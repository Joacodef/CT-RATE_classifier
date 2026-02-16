## Purpose

This file contains concise, actionable guidance for AI coding agents working on the CT 3D classifier repository. Keep responses, generated comments and code suggestions in English.

## Big picture (what this repo does)

- End-to-end training pipeline for multi-label pathology classification on 3D CT volumes.
- Preprocessing and caching use MONAI; training uses PyTorch; hyperparameter search uses Optuna; optional experiment logging with Weights & Biases (`wandb`).
- Main entry points: `scripts/train.py`, `scripts/inference.py`, `scripts/optimize_hyperparams.py`.

## Key files to read first

- `src/config/config.py` — YAML loader: substitutes `${ENV}` vars from `.env`, resolves paths into `Path` objects, and converts to `SimpleNamespace`.
- `configs/config.yaml` and `configs/config_example.yaml` — canonical experiment settings (model type, image_processing, cache, paths, pathologies list).
- `scripts/train.py` — CLI flags: `--config`, `--resume`, `--model-type`, `--model-variant`, `--workflow` and how run directories are derived.
- `scripts/inference.py` — CLI flags: `--config`, `--model`, `--input`, `--output`, `--device`, and the MONAI pipeline used for inference.
- `scripts/optimize_hyperparams.py` — Optuna staged optimization and how subset CSVs are selected for early trials.
- `src/training/trainer.py` — Core orchestration: caching strategy, DataLoader settings, model creation (`create_model`), loss selection, wandb init, and training loop.
- `src/data/dataset.py` — Dataset wrappers: `CTMetadataDataset`, `PersistentDataset`/`CacheDataset` usage, `LabelAttacherDataset` and `ApplyTransforms` patterns.

## Project-specific conventions and gotchas

- Configuration-driven: always load a YAML via `--config`. The loader substitutes environment variables from the repository root `.env`; required env vars (e.g., `DATA_DIR`, `IMG_DIR`, `CACHE_DIR`, `HF_TOKEN`) must exist or loading will raise an error.
- Cache invalidation: preprocessing transforms live under `image_processing` in the config — changing `target_spacing`/`target_shape_dhw` or other transform parameters will produce a new cache directory (the cache is keyed by a deterministic hash in `src/data/cache_utils.py`).
- Directory structure modes: `paths.dir_structure` supports `nested` and `flat` — `CTMetadataDataset` uses this to compute file paths (see `src/data/utils.py`).
- Model options: `model.type` ∈ {`resnet3d`, `densenet3d`, `vit3d`} with `model.variant` controlling sizes (e.g., `tiny`, `small`, `base`, or numeric variants like `121`).
- Loss: `loss_function.type` chooses between `FocalLoss` and `BCEWithLogitsLoss`. For BCE, positive class weights are computed automatically from the training dataframe in `trainer.py`.
- Resume semantics: `--resume` without path finds the latest checkpoint in the split-specific output directory derived from `config.paths.data_subsets.train`. Passing `--resume <path>` uses that file.
- HPO staged optimization: the Optuna script expects subset CSVs in the splits dir (`train_05_percent.csv`, `train_20_percent.csv`, `train_50_percent.csv`). If absent, optimization falls back to full data.

## Common developer workflows (examples)

All commands assume repository root. Example PowerShell usage:

```
# Run unit & integration tests
pytest .\tests\

# Train using configured split files
python scripts\train.py --config configs\config.yaml

# Resume training using latest checkpoint
python scripts\train.py --config configs\config.yaml --resume

# Run inference on a directory
python scripts\inference.py --config configs\config.yaml --model output\default\run_YYYYMMDD-HHMMSS_resnet3d_tiny\best_model.pth --input data\volumes\ --output results\batch_results.csv

# Generate or rebuild cache (see scripts/cache_management)
python scripts\cache_management\generate_cache.py --config configs\config.yaml --num-workers 8

# Hyperparameter search (Optuna)
python scripts\optimize_hyperparams.py --config configs\config.yaml --n-trials 100 --study-name vit3d-optimization --storage-db vit3d.db
```

## What an AI agent should do first when making changes

1. Read `configs/config_example.yaml` and `src/config/config.py` to understand required env vars and default paths. Avoid changing `.env` contents in PRs.
2. Inspect `scripts/train.py` and `src/training/trainer.py` to see end-to-end behavior, assumptions about CSV splits and checkpoint naming (`best_model.pth`, `last_checkpoint.pth`).
3. When touching transforms or `image_processing`, add a note that cache will be invalidated and tests/CI may need the cache rebuilt.
4. When adding CLI flags, mirror existing parsing style (argparse in scripts) and update help strings.

## Tests and build

- Tests use pytest; run `pytest .\tests\` from repo root. Keep tests lightweight; many tests mock disk I/O and external downloads.

## Final rules for generated content

- All comments, commit messages and generated code comments must be in English.
- Prefer small, iterative changes with tests. If touching training data paths or caching, include instructions to reproduce cache generation.
- Be specially careful when modifying cache saving/loading logic as it could lead to the need for full cache rebuilds, which is time-consuming.

