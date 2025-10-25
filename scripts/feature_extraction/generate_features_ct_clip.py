# scripts/feature_extraction/generate_features.py
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import inspect
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config
from src.data.dataset import CTMetadataDataset
from src.data.transforms import get_preprocessing_transforms
from src.training.trainer import create_model
from src.utils.logging_config import setup_logging
from monai.data import DataLoader, Dataset
import importlib
from pathlib import Path
from typing import Optional


def load_ct_clip_model(checkpoint_path: str, ct_clip_repo_path: Optional[str] = None, device='cpu'):
    # If repo path provided, ensure package is importable (pip install -e is preferred)
    if ct_clip_repo_path:
        sys.path.insert(0, str(Path(ct_clip_repo_path).resolve()))

    # Try to import the package (CT_CLIP) or some known modules
    module = None
    for cand in ('CT_CLIP', 'ct_clip', 'CT_CLIP.model', 'CT_CLIP.models'):
        try:
            module = importlib.import_module(cand)
            break
        except Exception:
            continue
    if module is None:
        raise RuntimeError('Could not import CT_CLIP package; ensure pip install -e or provide correct repo path')

    # Try common factory names
    factory = None
    for name in ('build_model', 'create_model', 'get_model', 'load_model'):
        if hasattr(module, name):
            factory = getattr(module, name)
            break

    if factory is None:
        # fallback: search for nn.Module subclasses in module
        for nm in dir(module):
            attr = getattr(module, nm)
            
            if inspect.isclass(attr) and issubclass(attr, nn.Module):
                # instantiate (best-effort: try no-args)
                try:
                    model = attr()
                    break
                except Exception:
                    continue
        else:
            raise RuntimeError('Could not find model factory or module class to build CT-CLIP model')

    if factory:
        model = factory()  # if factory requires args, adjust to factory(**kwargs)

    # Load checkpoint state_dict
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # Common patterns: ckpt['model_state_dict'] or ckpt contains state keys
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        raise RuntimeError('Checkpoint does not contain model state_dict')

    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # Find visual encoder and projection
    visual = None
    for cand in ('visual_transformer', 'visual', 'image_encoder', 'vision'):
        if hasattr(model, cand):
            visual = getattr(model, cand)
            break
    projection = getattr(model, 'to_visual_latent', None)

    class Wrapper(nn.Module):
        def __init__(self, vis, proj=None):
            super().__init__()
            self.vis = vis
            self.proj = proj
        def forward(self, x):
            out = self.vis(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if self.proj is not None:
                try:
                    return self.proj(out)
                except Exception:
                    return out
            return out

    wrapper = Wrapper(visual, projection).to(device)
    return wrapper


def adapt_model_to_feature_extractor(model: nn.Module, model_type: str) -> nn.Module:
    """
    Adapts a trained model to act as a feature extractor by removing its
    final classification layer.

    Args:
        model (nn.Module): The model to modify.
        model_type (str): The type of the model (e.g., 'resnet3d', 'densenet3d', 'vit3d').

    Returns:
        nn.Module: The modified model.

    Raises:
        ValueError: If the model type is not supported.
    """
    if model_type in ["resnet3d", "densenet3d"]:
        # For ResNet and DenseNet, the classifier is typically named 'fc' or 'classifier'
        if hasattr(model, 'fc'):
            in_features = model.fc[-1].in_features
            model.fc = nn.Identity()
            logging.info(f"Replaced model.fc with Identity. Feature dim: {in_features}")
        elif hasattr(model, 'classifier'):
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            logging.info(f"Replaced model.classifier with Identity. Feature dim: {in_features}")
        else:
            raise ValueError("Could not find 'fc' or 'classifier' layer in the model.")
    elif model_type == "vit3d":
        # For Vision Transformer, the classifier is named 'head'
        in_features = model.head[-1].in_features
        model.head = nn.Identity()
        logging.info(f"Replaced model.head with Identity. Feature dim: {in_features}")
    else:
        raise ValueError(f"Unsupported model type for feature extraction: {model_type}")
    
    return model


@torch.no_grad()
def generate_features(config, model_checkpoint: str, output_dir: Path, split: str, use_ct_clip: bool = False, ct_clip_repo_path: str = None, dry_run: bool = False):
    """
    Generates and saves feature vectors for a given dataset split.

    Args:
        config: The project configuration object.
        model_checkpoint (str): Path to the pre-trained model checkpoint.
        output_dir (Path): Directory to save the feature vectors.
        split (str): The data split to process ('train', 'valid', or 'all').
    """
    # 1. Setup device and logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    # 2. Load and adapt the model
    logger.info(f"Loading base model from checkpoint: {model_checkpoint}")

    if use_ct_clip:
        # Attempt to load the checkpoint and extract a visual encoder or serialized model
        ckpt = torch.load(model_checkpoint, map_location=device, weights_only=False)

        # If checkpoint is a dict and contains full model state, prefer to wrap it if serialized model object exists
        model = None
        if not ct_clip_repo_path:
            # If the checkpoint contains a serialized model object (not just state_dict), try to load it directly
            # Some checkpoints saved the whole model object; try that
            if not isinstance(ckpt, dict) and hasattr(ckpt, 'state_dict'):
                model = ckpt

        # If the checkpoint is a dict, check for visual-related keys to confirm presence of image encoder
        if isinstance(ckpt, dict):
            visual_keys = [k for k in ckpt.keys() if k.lower().startswith('visual_transformer') or 'visual_transformer' in k.lower() or any(x in k.lower() for x in ['to_visual_latent', 'to_pixels', 'to_patch_emb', 'visual', 'vision'])]
            if visual_keys:
                logger.info(f"Detected visual encoder keys in checkpoint (sample): {visual_keys[:10]}")
            else:
                logger.warning("No visual encoder keys detected in checkpoint top-level. If the image encoder was saved under a different prefix or the checkpoint is partial, you may need the CT-CLIP repo to reconstruct the model.")

        # If user provided a CT-CLIP repo path, try to import its factory and build the model there
        ct_clip_model = None
        # Always import and instantiate with correct parameters
        try:
            from ct_clip import CTCLIP
            from transformer_maskgit import CTViT
            from transformers import BertModel
            ct_clip_model = CTCLIP(
                image_encoder=CTViT(
                    dim=512,
                    codebook_size=8192,
                    image_size=480,
                    patch_size=20,
                    temporal_patch_size=10,
                    spatial_depth=4,
                    temporal_depth=4,
                    dim_head=32,
                    heads=8
                ),
                text_encoder=BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized"),
                dim_text=768,
                dim_image=294912,
                dim_latent=512,
                extra_latent_projection=False,
                use_mlm=False,
                downsample_image_embeds=False,
                use_all_token_embeds=False
            )
            logger.info("Instantiated CTCLIP model with correct parameters.")
        except Exception as e:
            logger.error(f"Failed to instantiate CTCLIP model: {e}")
            raise

        # If we have a ct_clip_model object, try loading the checkpoint state_dict into it
        if ct_clip_model is not None:
            try:
                # attempt to find model_state_dict in checkpoint
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    state = ckpt['model_state_dict']
                elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    state = ckpt
                else:
                    state = None

                if state is not None:
                    # Try loading whole model first (non-strict) as a best-effort
                    try:
                        ct_clip_model.load_state_dict(state, strict=False)
                        model = ct_clip_model
                        logger.info("Loaded checkpoint state into CT-CLIP model (non-strict).")
                    except Exception as e_full:
                        logger.warning(f"Full model load failed: {e_full}")
                        # FALLBACK: try to load only the visual submodule from checkpoint
                        visual_loaded = False
                        prefixes = ['visual_transformer', 'visual', 'image_encoder', 'vision']
                        for pref in prefixes:
                            try:
                                visual_state = {k[len(pref)+1:]: v for k, v in state.items() if k.startswith(pref + '.')}
                                if not visual_state:
                                    continue
                                if hasattr(ct_clip_model, pref):
                                    vis_mod = getattr(ct_clip_model, pref)
                                    # attempt to load visual-only parameters
                                    try:
                                        missing, unexpected = vis_mod.load_state_dict(visual_state, strict=False)
                                        logger.info(f"Loaded visual submodule from checkpoint under prefix '{pref}' (missing: {len(missing)}, unexpected: {len(unexpected)})")
                                        # build a small wrapper around the visual submodule and optional projection
                                        proj = getattr(ct_clip_model, 'to_visual_latent', None)
                                        class CTClipVisualWrapper(nn.Module):
                                            def __init__(self, vis, proj=None):
                                                super().__init__()
                                                self.vis = vis
                                                self.proj = proj
                                            def forward(self, x):
                                                out = self.vis(x)
                                                if isinstance(out, (list, tuple)):
                                                    out = out[0]
                                                if self.proj is not None:
                                                    try:
                                                        return self.proj(out)
                                                    except Exception:
                                                        return out
                                                return out
                                        model = CTClipVisualWrapper(vis_mod, proj).to(device)
                                        model.eval()
                                        visual_loaded = True
                                        break
                                    except Exception as e_vis:
                                        logger.warning(f"Failed loading visual submodule for prefix '{pref}': {e_vis}")
                            except Exception:
                                continue

                        if not visual_loaded:
                            logger.warning("Visual-only loading attempts did not succeed.")
                else:
                    logger.warning("No suitable state_dict found in checkpoint to load into CT-CLIP model.")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint into CT-CLIP model: {e}")

        # If we still don't have a model object but the checkpoint contains a serialized model, try to use it
        if model is None and isinstance(ckpt, nn.Module):
            model = ckpt

        # Final guard: if no model object, we cannot proceed automatically without CT-CLIP repo
        if model is None:
            logger.error("Could not construct a CT-CLIP model object from the checkpoint alone.\nProvide --ct-clip-repo-path pointing to the CT-CLIP repo (or pip install it) so the model class can be instantiated and the checkpoint state_dict loaded.\nAlternatively, provide a checkpoint that contains a serialized model object.")
            raise RuntimeError("CT-CLIP model object construction failed. See logs for details.")

        # use the ct-clip model as the feature extractor; find a visual submodule or projection
        feature_extractor = None
        for candidate in ['visual_transformer', 'visual', 'image_encoder', 'vision']:
            if hasattr(model, candidate):
                feature_extractor = getattr(model, candidate)
                logger.info(f"Using model.{candidate} as feature extractor.")
                break

        # If a projection layer exists (to_visual_latent), prefer it
        projection = None
        if hasattr(model, 'to_visual_latent'):
            projection = getattr(model, 'to_visual_latent')
        elif 'to_visual_latent.weight' in (ckpt.keys() if isinstance(ckpt, dict) else []):
            # projection weights exist in checkpoint but the model object may expose a different name
            projection = None

        # Wrap into a callable model for consistent interface
        if feature_extractor is not None and model is None:
            class CTClipWrapper(nn.Module):
                def __init__(self, feat_mod, proj_mod=None):
                    super().__init__()
                    self.feat = feat_mod
                    self.proj = proj_mod

                def forward(self, x):
                    out = self.feat(x)
                    # if out is tuple/list, try to pick first tensor
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if self.proj is not None:
                        # if projection is Linear-like
                        try:
                            return self.proj(out)
                        except Exception:
                            return out
                    return out

            model = CTClipWrapper(feature_extractor, projection).to(device)
            model.eval()
        else:
            # fallback: use model directly (if we have a full model instance)
            if model is not None:
                model = model.to(device)
                model.eval()
            else:
                logger.error("No model or feature extractor was constructed. Cannot proceed.")
                raise RuntimeError("CT-CLIP model object construction failed. See logs for details.")

    else:
        model = create_model(config)
        
        checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = adapt_model_to_feature_extractor(model, config.model.type).to(device)
        model.eval()

    # 3. Load the data
    logger.info(f"Loading data for '{split}' split...")
    if split == 'all':
        df_volumes = pd.read_csv(Path(config.paths.data_dir) / config.paths.full_dataset_csv)
    elif split == 'train':
        df_volumes = pd.read_csv(config.paths.data_subsets.train)
    elif split == 'valid':
        df_volumes = pd.read_csv(config.paths.data_subsets.valid)
    else:
        raise ValueError(f"Invalid split specified: {split}. Choose 'train', 'valid', or 'all'.")

    logger.info(f"Found {len(df_volumes)} volumes to process for the '{split}' split.")


    # --- Only process volumes without existing features ---
    import os
    checkpoint_name = os.path.splitext(os.path.basename(model_checkpoint))[0]
    features_root = output_dir
    model_features_dir = features_root / checkpoint_name / split
    model_features_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving features to: {model_features_dir}")


    # Efficiently build set of existing feature file base names (no extension)
    existing_features = set(
        os.path.splitext(f)[0]
        for f in os.listdir(model_features_dir)
        if f.endswith('.pt')
    )

    # Clean volume names in the DataFrame
    clean_names = df_volumes['VolumeName'].str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)
    mask = ~clean_names.isin(existing_features)
    skipped = (~mask).sum()
    to_process = mask.sum()
    logger.info(f"{skipped} features already exist and will be skipped. {to_process} volumes to process.")

    if to_process == 0:
        logger.info("All features already exist. Nothing to do.")
        return

    df_volumes_to_process = df_volumes[mask].reset_index(drop=True)

    preprocess_transforms = get_preprocessing_transforms(config)
    base_dataset = CTMetadataDataset(
        dataframe=df_volumes_to_process,
        img_dir=config.paths.img_dir,
        path_mode=config.paths.dir_structure,
    )
    processed_dataset = Dataset(data=base_dataset, transform=preprocess_transforms)
    data_loader = DataLoader(
        processed_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
    )



    import json

    # --- Save minimal config JSON ---
    import numpy as np
    def to_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if hasattr(val, 'tolist') and callable(val.tolist):
            return val.tolist()
        return val

    minimal_config = {
        "model": {
            "type": getattr(config.model, 'type', 'unknown'),
            "params": getattr(config.model, 'params', {}),
            "checkpoint": os.path.basename(model_checkpoint)
        },
        "preprocessing": {
            "target_shape": to_serializable(getattr(config.image_processing, 'target_shape_dhw', None)),
            "target_spacing": to_serializable(getattr(config.image_processing, 'target_spacing', None)),
            "clip_hu_min": to_serializable(getattr(config.image_processing, 'clip_hu_min', None)),
            "clip_hu_max": to_serializable(getattr(config.image_processing, 'clip_hu_max', None)),
            "normalization": to_serializable(getattr(config.image_processing, 'normalization', None))
        },
        "split": split
    }
    config_json_path = features_root / checkpoint_name / f"config_{split}.json"
    with open(config_json_path, "w") as f:
        json.dump(minimal_config, f, indent=2)

    batch_size = config.training.batch_size

    from transformers import BertTokenizer
    tokenizer = None
    if use_ct_clip:
        tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)

    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Generating features for '{split}' split")):
        images = batch["image"].to(device)
        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(images)
        volume_names = df_volumes.iloc[start_idx:end_idx]['VolumeName'].tolist()

        if use_ct_clip:
            with torch.no_grad():
                text_tokens = tokenizer([""] * images.size(0), return_tensors="pt", padding="max_length", truncation=True, max_length=200).to(device)
                _, image_latents, _ = model(text_tokens, images, device=device, return_latents=True)
                features = image_latents
        else:
            features = model(images)
        features = features.cpu()

        for i, volume_name in enumerate(volume_names):
            clean_volume_name = volume_name.replace(".nii.gz", "").replace(".nii", "")
            output_path = model_features_dir / f"{clean_volume_name}.pt"
            if output_path.exists():
                logger.info(f"Skipping {clean_volume_name}: feature already exists.")
                continue
            feature_vector = features[i]
            torch.save(feature_vector, output_path)

        if dry_run:
            logger.info("Dry-run mode enabled: processed one batch, exiting early.")
            break

    logger.info("Feature generation complete.")
    # If in dry-run mode, do not continue beyond the first batch (handled in loop); this is a safe exit.


def main():
    """Main function to run the feature generation script."""
    parser = argparse.ArgumentParser(
        description="Generate feature vectors from a pre-trained 3D model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the generated feature vectors will be saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "all"],
        help="The data split to process.",
    )
    # CT-CLIP specific options
    parser.add_argument(
        "--use-ct-clip",
        action="store_true",
        help="Use a CT-CLIP checkpoint as the model source. Requires --model-checkpoint to point to the CT-CLIP .pt file.",
    )
    parser.add_argument(
        "--ct-clip-repo-path",
        type=str,
        default=None,
        help="Optional path to a cloned CT-CLIP repo. If provided the script will try to import CT-CLIP factory functions from it. If not provided, the checkpoint must contain a serialized model object.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a quick dry-run: load model and process only a single batch (useful for testing).",
    )
    args = parser.parse_args()

    # Setup logging
    # Primary log location: repository-level logs/ directory so errors are easy to find
    repo_root = Path(__file__).resolve().parents[2]
    repo_logs = repo_root / "logs"
    repo_logs.mkdir(parents=True, exist_ok=True)
    log_file_path = repo_logs / f"feature_generation_{args.split}.log"
    setup_logging(log_file=log_file_path)

    # Also ensure the requested output dir exists (features will be saved there)
    out_log_dir = Path(args.output_dir)
    out_log_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    
    # Override batch size for efficiency if needed
    config.training.batch_size = config.training.batch_size * 2
    logging.info(f"Using batch size of {config.training.batch_size} for feature extraction.")

    # Ensure uncaught exceptions are logged to the file
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        logger = logging.getLogger(__name__)
        logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))
        # Also write a full traceback to the log file explicitly
        tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write('\n===== UNCAUGHT EXCEPTION =====\n')
                f.write(tb)
                f.write('\n===== END TRACE =====\n')
        except Exception:
            pass

    sys.excepthook = _excepthook

    # Run generation inside try/except so we can log and persist exceptions
    try:
        if args.use_ct_clip:
            # Pass ct-clip-specific options to generate_features via kwargs
            generate_features(
                config,
                args.model_checkpoint,
                Path(args.output_dir),
                args.split,
                use_ct_clip=True,
                ct_clip_repo_path=args.ct_clip_repo_path,
                dry_run=args.dry_run,
            )
        else:
            generate_features(
                config,
                args.model_checkpoint,
                Path(args.output_dir),
                args.split,
                dry_run=args.dry_run,
            )
    except Exception as e:
        # Log full traceback to the configured logger and ensure it's appended to the log file
        logging.getLogger(__name__).exception("Feature generation failed with an exception")
        raise


if __name__ == "__main__":
    main()