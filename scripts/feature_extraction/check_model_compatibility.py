#!/usr/bin/env python3
"""
scripts/check_model_compatibility.py

Utility to inspect a PyTorch checkpoint (for CT-CLIP) and attempt a minimal
compatibility check with a local CT-CLIP repo (if provided).

Features:
- Optionally download a checkpoint from a URL (requires requests).
- Load checkpoint with torch.load(..., weights_only=False) and print top-level keys.
- Search state_dict for likely image/vision encoder keys and print tensor shapes.
- If a local CT-CLIP repo path is given, try to discover nn.Module classes inside
  it, instantiate them (best-effort) and attempt to load the checkpoint (strict=False).
- Attempt small dummy forwards (2D and 3D) when a model instance is available.

This script is diagnostic and conservative: it never mutates your repo and prints
clear next steps rather than making assumptions.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pkgutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception:
    requests = None

import torch
import torch.nn as nn


def download_file(url: str, out_path: Path) -> Path:
    if requests is None:
        raise RuntimeError("requests is required to download files. pip install requests")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint from: {url}\n  -> to: {out_path}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = resp.headers.get('content-length')
    total = int(total) if total is not None else None
    downloaded = 0
    with open(out_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"Downloaded {downloaded}/{total} bytes ({pct:.1f}%)", end='\r')
    print('\nDownload complete')
    return out_path


def load_checkpoint(path: Path) -> Any:
    print(f"Loading checkpoint: {path}")
    try:
        ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
        print("Loaded checkpoint successfully.")
        return ckpt
    except TypeError:
        # Older torch versions may not accept weights_only kwarg
        ckpt = torch.load(str(path), map_location='cpu')
        print("Loaded checkpoint (without weights_only flag).")
        return ckpt
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        raise


def is_tensor_like(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def summarize_checkpoint(ckpt: Any, max_keys: int = 40) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if isinstance(ckpt, dict):
        report['type'] = 'dict'
        # Top-level keys as a quick surface view
        report['top_level_keys'] = list(ckpt.keys())[:max_keys]
        # Also provide a flattened view (recursive) of nested dict keys if present
        def _flatten_keys(d: Dict[str, Any], prefix: str = '') -> Iterable[str]:
            for k, v in d.items():
                full = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                yield full
                if isinstance(v, dict):
                    # Recurse into nested dicts
                    for kk in _flatten_keys(v, full):
                        yield kk

        try:
            flat_keys = list(_flatten_keys(ckpt))
        except Exception:
            flat_keys = report['top_level_keys']

        report['flattened_keys_sample'] = flat_keys[:max_keys]

        # If it looks like a state_dict (tensors), summarize shapes
        candidate_state = None

        # Common container key for model weights in many checkpoints
        if 'model_state_dict' in ckpt:
            candidate_state = ckpt['model_state_dict']
        # Some checkpoints put weights under 'state_dict' or other names; check a few
        elif 'state_dict' in ckpt:
            candidate_state = ckpt['state_dict']
        # Heuristic: if most values at top-level are tensors, it's likely a state_dict
        elif all(is_tensor_like(v) for v in ckpt.values()):
            candidate_state = ckpt

        if isinstance(candidate_state, dict):
            # Collect (key, value) pairs from the candidate state_dict, supporting nested dicts.
            def _collect_state_items(d: Dict[str, Any], prefix: str = '') -> List[Tuple[str, Any]]:
                items: List[Tuple[str, Any]] = []
                for k, v in d.items():
                    full = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                    if isinstance(v, dict):
                        items.extend(_collect_state_items(v, full))
                    else:
                        items.append((full, v))
                return items

            all_state_items = []
            try:
                all_state_items = _collect_state_items(candidate_state)
            except Exception:
                # Fallback: iterate top-level if recursion fails
                all_state_items = [(k, v) for k, v in candidate_state.items()]

            # Build a small sample with shapes/types for human-readable output
            report['state_dict_sample'] = []
            for i, (k, v) in enumerate(all_state_items):
                if i >= 40:
                    break
                if isinstance(v, torch.Tensor):
                    report['state_dict_sample'].append((k, tuple(v.shape)))
                else:
                    report['state_dict_sample'].append((k, type(v).__name__))

            # Broaden the set of keywords to detect visual/image encoders.
            # Include common CT-CLIP-like prefixes and transformer-related names.
            keywords = ['visual', 'vision', 'image', 'img', 'encoder', 'backbone',
                        'to_visual', 'to_visual_latent', 'patch', 'patch_embed', 'to_pixels',
                        'conv', 'proj', 'projection', 'visual_transformer']

            # Exclude keys that obviously belong to the text transformer to avoid false positives.
            # Common text encoder prefixes/terms (e.g., text_transformer.*, embeddings, token ids).
            def _is_text_key(key: str) -> bool:
                kl = key.lower()
                # keys that start with text_transformer or text_
                if kl.startswith('text_transformer') or kl.startswith('text_'):
                    return True
                # embedding and tokenizer-related names
                if 'word_embeddings' in kl or 'position_embeddings' in kl or 'token_type' in kl or 'position_ids' in kl:
                    return True
                return False

            # Search both the state item keys and flattened checkpoint keys for matches
            likely = [k for k, _ in all_state_items if any(kw in k.lower() for kw in keywords)]
            likely += [k for k in flat_keys if any(kw in k.lower() for kw in keywords)]
            # Filter out obvious text-encoder keys to reduce false positives, but only if
            # other visual-specific candidates remain. If filtering removes all candidates,
            # fall back to the original list so items like `text_transformer.encoder.layer.*`
            # can still be inspected (some CT-CLIP checkpoints use surprising prefixes).
            visual_candidates = [k for k in likely if not _is_text_key(k)]
            if visual_candidates:
                likely = visual_candidates
            # Deduplicate and cap
            seen = []
            for k in likely:
                if k not in seen:
                    seen.append(k)
            report['likely_image_keys'] = seen[:200]

            # Build a richer detail list for each likely visual key: include shape and nearby siblings
            key_to_shape: Dict[str, Optional[Tuple[int, ...]]] = {}
            for k, v in all_state_items:
                if isinstance(v, torch.Tensor):
                    try:
                        key_to_shape[k] = tuple(v.shape)
                    except Exception:
                        key_to_shape[k] = None
                else:
                    key_to_shape[k] = None

            visual_details: List[Dict[str, Any]] = []
            for k in report['likely_image_keys']:
                shape = key_to_shape.get(k)
                # compute a sibling context: keys that share the same prefix up to last dot
                if '.' in k:
                    prefix = k.rsplit('.', 1)[0]
                    siblings = [s for s in key_to_shape.keys() if s.startswith(prefix + '.')][:8]
                else:
                    prefix = ''
                    siblings = [s for s in key_to_shape.keys() if '.' not in s][:8]
                visual_details.append({'key': k, 'shape': shape, 'prefix': prefix, 'siblings_sample': siblings})

            report['visual_key_details'] = visual_details
    else:
        report['type'] = type(ckpt).__name__
    return report


def discover_nn_modules(repo_path: Path) -> List[Tuple[str, str]]:
    """Discover possible nn.Module classes in a repo path.

    Returns list of (module_name, class_name).
    """
    found: List[Tuple[str, str]] = []
    repo_path = repo_path.resolve()
    if not repo_path.exists():
        print(f"Repo path does not exist: {repo_path}")
        return found

    # Add repo root to sys.path temporarily
    sys.path.insert(0, str(repo_path))
    try:
        # Walk packages under the repo directory
        for finder, name, ispkg in pkgutil.walk_packages([str(repo_path)]):
            try:
                full_name = name
                module = importlib.import_module(full_name)
            except Exception:
                # try importing as package.module
                continue
            for member_name, member in inspect.getmembers(module, inspect.isclass):
                try:
                    if issubclass(member, nn.Module) and member is not nn.Module:
                        found.append((module.__name__, member.__name__))
                except Exception:
                    continue
    finally:
        # Remove the inserted path
        try:
            sys.path.remove(str(repo_path))
        except Exception:
            pass
    return found


def try_instantiate(module_name: str, class_name: str) -> Optional[nn.Module]:
    print(f"Trying to instantiate {module_name}.{class_name}() with no args")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls()
        if isinstance(instance, nn.Module):
            print(f"Successfully instantiated {class_name}")
            return instance
    except Exception as e:
        print(f"Instantiation failed: {e}")
    return None


def attempt_load_state_dict(model: nn.Module, ckpt: Any) -> None:
    # Extract candidate state dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(is_tensor_like(v) for v in ckpt.values()):
        state = ckpt
    else:
        print("No obvious state_dict found at top-level of checkpoint to load.")
        return

    if not isinstance(state, dict):
        print("Found state object but it's not a dict. Skipping load attempt.")
        return

    print("Attempting to load state_dict into the model with strict=False (best-effort)")
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"load_state_dict raised an exception: {e}")
        return
    # PyTorch <=1.8 returns NamedTuple, newer returns dict; handle both
    try:
        if hasattr(missing, 'keys') or isinstance(missing, dict):
            print("Load returned a mapping result (PyTorch newer).")
            print(missing)
        else:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
    except Exception:
        print("Loaded state_dict; can't nicely summarize missing/unexpected keys for this torch version.")


def try_forward(model: nn.Module, shapes: Iterable[Tuple[int, ...]]) -> None:
    model_cpu = model.to('cpu')
    model_cpu.eval()
    for shape in shapes:
        x = torch.randn(*shape)
        try:
            with torch.no_grad():
                out = model_cpu(x)
            print(f"Forward OK for input shape {shape} -> output shape: {getattr(out, 'shape', type(out))}")
        except Exception as e:
            print(f"Forward failed for input shape {shape}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Check CT-CLIP checkpoint and try minimal compatibility checks.")
    parser.add_argument('--ct-clip-repo-path', type=str, default=None, help='Path to cloned CT-CLIP repo (optional).')
    parser.add_argument('--checkpoint-url', type=str, default=None, help='Optional URL to download the checkpoint (HF resolve URL).')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Local path to checkpoint .pt file.')
    parser.add_argument('--dummy-2d-shape', type=str, default='1,3,224,224', help='Comma-separated shape for a 2D dummy forward (e.g., 1,3,224,224)')
    parser.add_argument('--dummy-3d-shape', type=str, default='1,1,32,128,128', help='Comma-separated shape for a 3D dummy forward (e.g., 1,1,32,128,128)')
    parser.add_argument('--no-download', action='store_true', help='If set, do not attempt to download checkpoint even if URL provided.')
    args = parser.parse_args()

    ckpt_path: Optional[Path] = None

    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        if not ckpt_path.exists():
            print(f"Provided checkpoint path does not exist: {ckpt_path}")
            ckpt_path = None

    if ckpt_path is None and args.checkpoint_url and not args.no_download:
        # Download to temp
        tmp_dir = Path(tempfile.gettempdir()) / 'ct_clip_ckpt'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(args.checkpoint_url).name
        out_path = tmp_dir / filename
        try:
            download_file(args.checkpoint_url, out_path)
            ckpt_path = out_path
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")

    if ckpt_path is None:
        print("No checkpoint available. Provide --checkpoint-path or --checkpoint-url (and ensure HF auth if needed).")
    else:
        ckpt = load_checkpoint(ckpt_path)
        # Summarize checkpoint (but keep terminal output concise and focused on visual encoder candidates)
        summary = summarize_checkpoint(ckpt)
        print('\n=== Visual encoder candidates summary ===')

        # If the summarizer produced detailed visual_key_details, print those succinctly
        visual_details = summary.get('visual_key_details')
        if visual_details:
            print(f"Found {len(visual_details)} visual-related entries (showing up to 50).\n")
            for entry in visual_details[:50]:
                key = entry.get('key')
                shape = entry.get('shape')
                prefix = entry.get('prefix')
                siblings = entry.get('siblings_sample', [])
                print(f"- key: {key}")
                print(f"  shape: {shape}")
                print(f"  prefix: {prefix}")
                if siblings:
                    print(f"  siblings (sample):")
                    for s in siblings:
                        print(f"    {s}")
                print('')
        else:
            # Fallback: show likely_image_keys if available
            likely = summary.get('likely_image_keys', [])
            if likely:
                print(f"No detailed visual entries, but found {len(likely)} likely visual keys. Showing up to 100:")
                for k in likely[:100]:
                    print(f" - {k}")
            else:
                # Final fallback: print a short sample of top-level keys to help user inspect
                print("No visual-related keys detected. Top-level keys sample:")
                for k in summary.get('top_level_keys', [])[:50]:
                    print(f" - {k}")

    discovered = []
    if args.ct_clip_repo_path:
        repo_path = Path(args.ct_clip_repo_path)
        print(f"\nSearching for nn.Module classes in repo: {repo_path}")
        discovered = discover_nn_modules(repo_path)
        if discovered:
            print(f"Found {len(discovered)} candidate nn.Module classes (module, class):")
            for mod, cls in discovered[:50]:
                print(f" - {mod}.{cls}")
        else:
            print("No nn.Module subclasses discovered automatically in the provided repo path.")

    # If we discovered classes, try to instantiate the first few and attempt load/forward
    attempts = 0
    for mod, cls in discovered[:10]:
        inst = try_instantiate(mod, cls)
        attempts += 1
        if inst is None:
            continue
        if ckpt_path is not None:
            try:
                attempt_load_state_dict(inst, ckpt)
            except Exception as e:
                print(f"Error while trying to load state_dict: {e}")
        # Try forward with dummy shapes
        try:
            shapes = []
            shapes.append(tuple(int(x) for x in args.dummy_2d_shape.split(',')))
            shapes.append(tuple(int(x) for x in args.dummy_3d_shape.split(',')))
            try_forward(inst, shapes)
        except Exception as e:
            print(f"Error during forward attempts: {e}")
        # safety: break if too many attempts
        if attempts >= 3:
            break

    print('\nCompatibility check complete. Review the printed summary to decide next steps.')


if __name__ == '__main__':
    main()
