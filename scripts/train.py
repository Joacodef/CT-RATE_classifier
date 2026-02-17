# scripts/train.py
import sys
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import torch
import torch.multiprocessing

# Set the multiprocessing sharing strategy to 'file_system' for Linux systems.
# This is a workaround for the "Too many open files" error that can occur when
# using the default 'file_descriptor' strategy with a large number of workers.
if sys.platform == "linux":
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError as e:
        # This might fail if the strategy is already set, which is fine.
        logging.warning(
            "Failed to set multiprocessing sharing strategy, it might already be set. "
            f"Details: {e}"
        )

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

from src.config import load_config
from src.training.trainer import train_model
from src.utils.logging_config import setup_logging
from src.training.utils import find_latest_checkpoint


def _sanitize_component(component: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", component.strip())
    sanitized = sanitized.strip("-")
    return sanitized if sanitized else "run"


def _build_run_folder_name(config, fold: Optional[int] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [timestamp]

    model = getattr(config, "model", SimpleNamespace())
    if getattr(model, "type", None):
        parts.append(str(model.type))
    if getattr(model, "variant", None):
        parts.append(str(model.variant))

    workflow = getattr(config, "workflow", SimpleNamespace())
    workflow_mode = getattr(workflow, "mode", None)
    if workflow_mode and workflow_mode != "end-to-end":
        parts.append(str(workflow_mode))

    cleaned = [_sanitize_component(str(part)) for part in parts if part]
    name_stub = "_".join(cleaned) if cleaned else timestamp
    return f"run_{name_stub}"


def _find_latest_run_directory(parent_dir: Path) -> Optional[Path]:
    if not parent_dir.exists():
        return None

    try:
        run_dirs = [entry for entry in parent_dir.iterdir() if entry.is_dir() and entry.name.startswith("run_")]
    except FileNotFoundError:
        return None

    if not run_dirs:
        return None

    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _derive_split_name(config) -> str:
    try:
        data_subsets = getattr(config.paths, "data_subsets", None)
        if not data_subsets or not hasattr(data_subsets, "train"):
            return "default"
        train_path = Path(data_subsets.train)
        if train_path.parent.name:
            return train_path.parent.name
    except Exception:
        pass
    return "default"


# scripts/train.py

def main():
    """
    Main function to start or resume the training process.
    It loads a base configuration and allows for overrides via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the training pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        choices=['resnet3d', 'densenet3d', 'vit3d'],
        help='Override the model architecture from the config file.'
    )
    parser.add_argument(
        '--model-variant',
        type=str,
        default=None,
        help='Override the model variant from the config file.'
    )
    parser.add_argument(
        '--workflow',
        type=str,
        default=None,
        choices=['end-to-end'],
        help="Set the training workflow. Only 'end-to-end' is supported."
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        const=True,
        default=None,
        help=(
            "Resume training. \n"
            "- If used as a flag (`--resume`), it automatically finds the latest checkpoint.\n"
            "- If given a path (`--resume path/to/ckpt.pth`), it resumes from that specific file."
        )
    )
    args = parser.parse_args()

    # 1. Load the base configuration
    config = load_config(args.config)

    # 2. Override config with command-line arguments if provided
    if args.model_type:
        logging.info(f"Overriding model.type from '{config.model.type}' to '{args.model_type}'")
        config.model.type = args.model_type
    
    if args.model_variant:
        logging.info(f"Overriding model.variant from '{config.model.variant}' to '{args.model_variant}'")
        config.model.variant = args.model_variant

    # Workflow-level override (only --workflow is supported from CLI)
    if args.workflow is not None:
        prev_workflow = getattr(config, 'workflow', SimpleNamespace(mode='end-to-end'))
        logging.info(f"Overriding workflow.mode from '{getattr(prev_workflow, 'mode', None)}' to '{args.workflow}'")
        if not hasattr(config, 'workflow'):
            config.workflow = SimpleNamespace()
        config.workflow.mode = args.workflow

    if getattr(config, 'workflow', None) and getattr(config.workflow, 'mode', 'end-to-end') != 'end-to-end':
        logging.error("Only 'end-to-end' workflow is supported.")
        return

    # 3. Prepare the output directory structure with timestamped run folders.
    base_output_dir = Path(config.paths.output_dir)
    split_name_raw = _derive_split_name(config)

    resume_checkpoint_path: Optional[Path] = None
    run_output_dir: Optional[Path] = None

    if args.resume:
        if args.resume is True:
            latest_run_dir = _find_latest_run_directory(base_output_dir)
            search_dir = latest_run_dir if latest_run_dir else base_output_dir
            logging.info(f"Attempting automatic resume. Looking for checkpoints in {search_dir}...")
            if search_dir.exists():
                latest_checkpoint = find_latest_checkpoint(search_dir)
                if latest_checkpoint:
                    run_output_dir = latest_checkpoint.parent
                    resume_checkpoint_path = latest_checkpoint
                    logging.info(f"Auto-resume: using checkpoint {latest_checkpoint}")
                else:
                    logging.warning(
                        f"Automatic resume failed: no checkpoint found in {search_dir}. "
                        "Starting a new training session."
                    )
            else:
                logging.warning(
                    f"Automatic resume failed: search directory {search_dir} does not exist. "
                    "Starting a new training session."
                )
        else:
            resume_candidate = Path(args.resume).expanduser()
            logging.info(f"Attempting to resume using provided path: {resume_candidate}")
            if not resume_candidate.exists():
                logging.error(f"Resume failed: specified path does not exist: {resume_candidate}")
                return
            if resume_candidate.is_file():
                run_output_dir = resume_candidate.parent
                resume_checkpoint_path = resume_candidate
            elif resume_candidate.is_dir():
                run_output_dir = resume_candidate
                latest_checkpoint = find_latest_checkpoint(run_output_dir)
                if latest_checkpoint:
                    resume_checkpoint_path = latest_checkpoint
                else:
                    logging.warning(
                        f"No checkpoint found inside {run_output_dir}. Training will start from scratch in this directory."
                    )
            else:
                logging.error(
                    f"Resume failed: {resume_candidate} is neither a file nor a directory."
                )
                return

    if run_output_dir is None:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        run_folder_name = _build_run_folder_name(config)
        run_output_dir = base_output_dir / run_folder_name

    run_output_dir.mkdir(parents=True, exist_ok=True)

    config.paths.base_output_dir = base_output_dir.resolve()
    config.paths.split_name = split_name_raw
    config.paths.split_output_dir = base_output_dir.resolve()
    config.paths.fold_output_dir = base_output_dir.resolve()
    config.paths.output_dir = run_output_dir.resolve()
    config.paths.run_name = run_output_dir.name

    if resume_checkpoint_path:
        config.training.resume_from_checkpoint = str(resume_checkpoint_path)
        logging.info(f"Training will resume from checkpoint: {resume_checkpoint_path}")
    elif hasattr(config.training, 'resume_from_checkpoint'):
        config.training.resume_from_checkpoint = None

    logger.info(f"Output will be saved to: {config.paths.output_dir}")
    workflow_ns = getattr(config, 'workflow', None)
    current_mode = getattr(workflow_ns, 'mode', None) if workflow_ns else None
    logging.info(
        "Current workflow config before training: mode=%s",
        current_mode,
    )

    # 5. Setup logging and start training
    log_dir = Path(config.paths.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_dir / 'training.log')
    
    logging.info("Configuration loaded and processed. Starting training.")
    train_model(config)

if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility in multiprocessing.
    # 'spawn' is required for using CUDA in subprocesses.
    # torch.multiprocessing.set_start_method('spawn', force=True)
    main()