import sys
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import scripts.train as train_mod


def test_sanitization_and_build_name():
    cfg = SimpleNamespace()
    # include weird chars in type/variant and a non-default workflow mode
    cfg.model = SimpleNamespace(type="My/Model<>", variant="v:1")
    cfg.workflow = SimpleNamespace(mode="feature-based")

    # Build name must start with run_ and contain sanitized tokens (no spaces or forbidden chars)
    name = train_mod._build_run_folder_name(cfg, fold=None)
    assert name.startswith("run_")
    # sanitized should replace slash/colon/angle with hyphens and not contain spaces
    assert " " not in name
    assert "/" not in name and ":" not in name and "<" not in name
    assert "My-Model" in name or "My-Model" in name.replace("_", "_")


def test_find_latest_run_directory_respects_mtime(tmp_path: Path):
    parent = tmp_path / "parent"
    parent.mkdir()

    older = parent / "run_old"
    newer = parent / "run_new"
    older.mkdir()
    newer.mkdir()

    # set mtimes explicitly: newer gets larger mtime
    older_file = older / ".dummy"
    newer_file = newer / ".dummy"
    older_file.write_text("x")
    newer_file.write_text("y")

    # older: now - 100 seconds, newer: now
    now = int(os.path.getmtime(newer_file))
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    found = train_mod._find_latest_run_directory(parent)
    assert found is not None
    assert found.resolve() == newer.resolve()


def test_cli_resume_auto_selects_latest_and_windows_feature_overrides(monkeypatch, tmp_path: Path):
    # Prepare config values
    out = tmp_path / "output"
    out.mkdir()
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    train_csv = splits_dir / "train.csv"
    train_csv.write_text("a,b,c")

    # Feature dir with train/valid layout
    feat_root = tmp_path / "features"
    train_sub = feat_root / "train"
    valid_sub = feat_root / "valid"
    train_sub.mkdir(parents=True)
    valid_sub.mkdir()
    # write a dummy feature file
    ckpt_file = train_sub / "sample.pt"
    ckpt_file.write_text("fake")

    cfg = SimpleNamespace()
    cfg.paths = SimpleNamespace(output_dir=str(out), data_subsets=SimpleNamespace(train=str(train_csv)))
    cfg.workflow = SimpleNamespace(mode="feature-based", feature_config=SimpleNamespace(feature_dir=str(feat_root)))
    cfg.training = SimpleNamespace(augment=True, num_workers=4, pin_memory=True)
    cfg.model = SimpleNamespace(type="mlp", variant=None)

    # create two run dirs under out/splits/full so that the latest is chosen
    split_dir = out / splits_dir.name
    fold_dir = split_dir / "full"
    (fold_dir / "run_old").mkdir(parents=True)
    run_latest = fold_dir / "run_latest"
    run_latest.mkdir(parents=True)

    # touch a checkpoint file inside run_latest
    latest_ckpt = run_latest / "best_checkpoint.pth"
    latest_ckpt.write_text("ckpt")

    # monkeypatch load_config to return our cfg
    monkeypatch.setattr(train_mod, "load_config", lambda path: cfg)

    # monkeypatch find_latest_checkpoint (imported symbol in module) to return our created checkpoint
    monkeypatch.setattr(train_mod, "find_latest_checkpoint", lambda search_dir: latest_ckpt)

    # ensure setup_logging / train_model do not actually run
    monkeypatch.setattr(train_mod, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(train_mod, "train_model", lambda config: None)

    # Ensure the filesystem-based latest-run finder will prefer run_latest by mtime
    # Set run_latest mtime > run_old
    old_dir = fold_dir / "run_old"
    now = int(os.path.getmtime(run_latest))
    os.utime(old_dir, (now - 100, now - 100))
    os.utime(run_latest, (now, now))

    # Simulate Windows platform so feature-based adjustments happen
    monkeypatch.setattr(sys, "platform", "win32", raising=False)

    # Run main with --config and --resume flag
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "dummy_config.yaml", "--resume"])

    # Execute main (should return cleanly because we've stubbed training)
    train_mod.main()

    # After main, config.training should have been modified: augment disabled, num_workers set to 0, pin_memory False
    assert cfg.training.augment is False
    assert cfg.training.num_workers == 0
    # pin_memory may or may not exist, but if it exists, it should be False
    if hasattr(cfg.training, "pin_memory"):
        assert cfg.training.pin_memory is False

    # resume_from_checkpoint should be set to the string path of our latest checkpoint
    assert hasattr(cfg.training, "resume_from_checkpoint")
    assert str(latest_ckpt) in cfg.training.resume_from_checkpoint
