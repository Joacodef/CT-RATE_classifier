# scripts/cache_management/generate_cache.py

import argparse
import atexit
import faulthandler
import functools
import json
import logging
import os
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import resource
from pathlib import Path
from threading import current_thread
from datetime import datetime, timezone

import pandas as pd
import torch
from tqdm import tqdm

# --- MONAI Imports ---
from monai.data import DataLoader, PersistentDataset

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config
from src.data.cache_utils import (
    deterministic_hash,
    get_or_create_cache_subdirectory,
    worker_init_fn
)
from src.data.dataset import CTMetadataDataset
from src.data.transforms import get_preprocessing_transforms
from src.data.utils import get_dynamic_image_path
from scripts.data_preparation.verify_and_download import download_worker

# --- Logging Configuration ---
logger = logging.getLogger("generate_cache")
CACHE_DATALOADER_PREFETCH_FACTOR = 1


def flush_log_handlers():
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def write_runtime_status(log_directory: Path, payload: dict):
    status_file = log_directory / "generate_cache_status.json"
    safe_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with status_file.open("w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2, ensure_ascii=False)


def get_process_rss_mb() -> float | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = float(usage.ru_maxrss)
        return rss_kb / 1024.0
    except Exception:
        return None


def get_system_available_mb() -> float | None:
    try:
        meminfo_path = Path("/proc/meminfo")
        if not meminfo_path.exists():
            return None
        with meminfo_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        return None
    return None


def get_memory_snapshot() -> dict:
    rss_mb = get_process_rss_mb()
    available_mb = get_system_available_mb()
    snapshot = {}
    if rss_mb is not None:
        snapshot["process_rss_mb"] = round(rss_mb, 2)
    if available_mb is not None:
        snapshot["system_available_mb"] = round(available_mb, 2)
    return snapshot


def log_memory_snapshot(context: str):
    snapshot = get_memory_snapshot()
    if snapshot:
        logger.info(f"Memory snapshot [{context}]: {snapshot}")


def install_crash_hooks(log_directory: Path):
    fatal_log_path = log_directory / "generate_cache_fatal.log"
    fatal_log_handle = fatal_log_path.open("a", encoding="utf-8")

    faulthandler.enable(file=fatal_log_handle, all_threads=True)

    def _handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("Interrupted by user (KeyboardInterrupt).")
        else:
            logger.critical("Uncaught exception in main thread", exc_info=(exc_type, exc_value, exc_traceback))
        flush_log_handlers()

    def _thread_exception_handler(args):
        logger.critical(
            f"Uncaught exception in thread '{args.thread.name if args.thread else 'unknown'}'",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        flush_log_handlers()

    sys.excepthook = _handle_exception
    if hasattr(sys, "unraisablehook"):
        original_unraisablehook = sys.unraisablehook

        def _unraisable_hook(unraisable):
            logger.critical(
                f"Unraisable exception in object {unraisable.object}",
                exc_info=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback),
            )
            flush_log_handlers()
            original_unraisablehook(unraisable)

        sys.unraisablehook = _unraisable_hook

    try:
        import threading
        threading.excepthook = _thread_exception_handler
    except Exception:
        logger.warning("Could not install threading.excepthook.")

    def _signal_handler(signum, _frame):
        signal_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
        logger.critical(f"Received termination signal: {signal_name} ({signum}).")
        flush_log_handlers()

    for sig_name in ["SIGTERM", "SIGINT", "SIGABRT", "SIGBREAK"]:
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _signal_handler)
            except Exception:
                logger.warning(f"Could not register handler for signal {sig_name}.")

    def _on_exit():
        logger.info("Process exiting. Flushing logs.")
        flush_log_handlers()
        try:
            fatal_log_handle.flush()
            fatal_log_handle.close()
        except Exception:
            pass

    atexit.register(_on_exit)


def setup_logging(log_directory: Path):
    log_directory.mkdir(parents=True, exist_ok=True)

    log_file_path = log_directory / "generate_cache.log"
    error_log_file_path = log_directory / "generate_cache_errors.log"
    missing_log_file_path = log_directory / "generate_cache_missing_files.log"

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    info_file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)

    error_file_handler = logging.FileHandler(error_log_file_path, mode='a', encoding='utf-8')
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    missing_file_handler = logging.FileHandler(missing_log_file_path, mode='a', encoding='utf-8')
    missing_file_handler.setLevel(logging.WARNING)
    missing_file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(info_file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(missing_file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized in directory: {log_directory}")
    install_crash_hooks(log_directory)
    write_runtime_status(
        log_directory,
        {
            "status": "started",
            "pid": os.getpid(),
            "script": "generate_cache.py",
            **get_memory_snapshot(),
        },
    )

torch.load = functools.partial(torch.load, weights_only=False)

class CachingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform, cache_dir):
        self.persistent_ds = PersistentDataset(
            data=base_dataset, transform=transform,
            cache_dir=cache_dir, hash_func=deterministic_hash
        )
        self.base_ds = base_dataset

    def __len__(self) -> int:
        return len(self.persistent_ds)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self.persistent_ds[idx]
        except Exception as e:
            volume_name = self.base_ds[idx]['volume_name']
            return {"VolumeName": volume_name, "error": e}

def identity_collate(batch):
    return batch


def is_worker_termination_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return (
        "dataloader worker" in error_text
        and ("exited unexpectedly" in error_text or "killed by signal" in error_text)
    )


def build_worker_retry_schedule(initial_num_workers: int) -> list[int]:
    workers = max(0, int(initial_num_workers))
    schedule = []
    while workers > 0:
        schedule.append(workers)
        if workers == 1:
            break
        workers = max(1, workers // 2)
    if 0 not in schedule:
        schedule.append(0)
    return schedule


def cache_batch_with_retries(caching_ds, base_ds, requested_num_workers: int, batch_num: int):
    worker_schedule = build_worker_retry_schedule(requested_num_workers)
    last_exception = None

    for attempt_idx, attempt_workers in enumerate(worker_schedule, start=1):
        logger.info(
            f"[Batch {batch_num}] Cache attempt {attempt_idx}/{len(worker_schedule)} "
            f"with DataLoader num_workers={attempt_workers}."
        )
        log_memory_snapshot(f"batch_{batch_num}_attempt_{attempt_idx}_start")

        data_loader = DataLoader(
            caching_ds,
            batch_size=None,
            num_workers=attempt_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=identity_collate,
            **({"prefetch_factor": CACHE_DATALOADER_PREFETCH_FACTOR} if attempt_workers > 0 else {}),
        )

        if attempt_workers > 0:
            logger.info(
                f"[Batch {batch_num}] DataLoader prefetch_factor set to {CACHE_DATALOADER_PREFETCH_FACTOR}."
            )

        successfully_cached_volumes = set()
        try:
            progress_bar = tqdm(
                enumerate(data_loader),
                desc=f"Caching Batch {batch_num} (workers={attempt_workers})",
                total=len(caching_ds),
            )
            for j, item in progress_bar:
                if item and isinstance(item, dict) and 'error' not in item:
                    original_volume_name = base_ds[j]['volume_name']
                    successfully_cached_volumes.add(original_volume_name)
                elif item and isinstance(item, dict) and 'error' in item:
                    volume_name = item.get('VolumeName', base_ds[j]['volume_name'])
                    logger.error(f"[Batch {batch_num}] Cache generation failed for {volume_name}: {item['error']}")

            return successfully_cached_volumes, attempt_workers
        except RuntimeError as exc:
            last_exception = exc
            if is_worker_termination_error(exc):
                has_next_attempt = attempt_idx < len(worker_schedule)
                logger.error(
                    f"[Batch {batch_num}] DataLoader worker failure at num_workers={attempt_workers}: {exc}"
                )
                log_memory_snapshot(f"batch_{batch_num}_attempt_{attempt_idx}_worker_failure")
                if has_next_attempt:
                    next_workers = worker_schedule[attempt_idx]
                    logger.warning(
                        f"[Batch {batch_num}] Retrying cache with reduced num_workers={next_workers}."
                    )
                    flush_log_handlers()
                    continue

            raise

    if last_exception is not None:
        raise last_exception
    return set(), requested_num_workers

def analyze_cache_state(volumes_df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    if not cache_dir.exists():
        logger.info("Cache directory does not exist. All volumes are missing.")
        return volumes_df
    
    missing_volumes = []
    logger.info(f"Analyzing cache state in: {cache_dir}")
    for _, row in tqdm(volumes_df.iterrows(), total=len(volumes_df), desc="Checking cache status"):
        data_item = {"volume_name": row['VolumeName']}
        hashed_filename_str = deterministic_hash(data_item).decode('utf-8')
        cache_filepath = cache_dir / f"{hashed_filename_str}.pt"
        if not cache_filepath.exists():
            missing_volumes.append(row.to_dict())
    return pd.DataFrame(missing_volumes) if missing_volumes else pd.DataFrame()

def process_in_batches(config, files_df: pd.DataFrame, batch_size: int, num_workers: int):
    num_batches = (len(files_df) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(files_df)} files in {num_batches} batches of size {batch_size}.")

    preprocess_transforms = get_preprocessing_transforms(config)
    cache_dir = get_or_create_cache_subdirectory(
        Path(config.paths.cache_dir), preprocess_transforms, split="unified"
    )
    logs_dir = Path(config.paths.logs_dir)
    
    # Create a single parent directory for all temporary batch caches
    main_temp_hf_dir = Path(config.paths.data_dir) / "temp_hf_batch_cache"
    main_temp_hf_dir.mkdir(exist_ok=True)

    try:
        for i in range(num_batches):
            batch_num = i + 1
            logger.info(f"\n--- Starting Batch {batch_num}/{num_batches} ---")
            log_memory_snapshot(f"batch_{batch_num}_start")
            write_runtime_status(
                logs_dir,
                {
                    "status": "running",
                    "stage": "batch_start",
                    "batch": batch_num,
                    "num_batches": num_batches,
                    "total_files": len(files_df),
                    **get_memory_snapshot(),
                },
            )
            
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_df = files_df.iloc[start_idx:end_idx]

            # Create a dedicated temporary cache for this specific batch
            batch_hf_cache_dir = main_temp_hf_dir / f"batch_{batch_num}"
            batch_hf_cache_dir.mkdir(exist_ok=True)

            volumes_to_download = [
                row['VolumeName'] for _, row in batch_df.iterrows()
                if not get_dynamic_image_path(Path(config.paths.img_dir), row['VolumeName'], config.paths.dir_structure).exists()
            ]
            
            downloaded_volume_names = set()
            failed_downloads = []
            if volumes_to_download:
                logger.info(f"[Batch {batch_num}] Downloading {len(volumes_to_download)} files...")
                with ThreadPoolExecutor(max_workers=config.downloads.max_workers) as executor:
                    task = functools.partial(download_worker, config=config, hf_cache_dir=batch_hf_cache_dir)
                    results = list(tqdm(executor.map(task, volumes_to_download), total=len(volumes_to_download), desc=f"Downloading Batch {batch_num}"))
                for vol_name, res in zip(volumes_to_download, results):
                    if "OK" in res:
                        downloaded_volume_names.add(vol_name)
                    else:
                        failed_downloads.append((vol_name, res))
                        logger.error(f"[Batch {batch_num}] Download failed for {vol_name}: {res}")

                unresolved_volumes = [
                    volume_name for volume_name in volumes_to_download
                    if not get_dynamic_image_path(Path(config.paths.img_dir), volume_name, config.paths.dir_structure).exists()
                ]
                if unresolved_volumes:
                    logger.warning(
                        f"[Batch {batch_num}] {len(unresolved_volumes)} file(s) still missing after download attempts. "
                        f"See generate_cache_missing_files.log for details."
                    )
                    for volume_name in unresolved_volumes:
                        logger.warning(f"[Batch {batch_num}] Missing raw file: {volume_name}")
            else:
                logger.info(f"[Batch {batch_num}] All raw files for this batch are already present locally. No downloads needed.")

            logger.info(f"[Batch {batch_num}] Caching {len(batch_df)} files...")
            base_ds = CTMetadataDataset(
                dataframe=batch_df, img_dir=Path(config.paths.img_dir),
                path_mode=config.paths.dir_structure
            )
            caching_ds = CachingDataset(
                base_dataset=base_ds, transform=preprocess_transforms,
                cache_dir=cache_dir
            )
            successfully_cached_volumes, final_workers_used = cache_batch_with_retries(
                caching_ds=caching_ds,
                base_ds=base_ds,
                requested_num_workers=num_workers,
                batch_num=batch_num,
            )
            logger.info(
                f"[Batch {batch_num}] Cache completed with num_workers={final_workers_used}. "
                f"Successfully cached {len(successfully_cached_volumes)} volume(s)."
            )

            if failed_downloads:
                logger.warning(
                    f"[Batch {batch_num}] {len(failed_downloads)} download(s) failed. "
                    f"Review generate_cache_errors.log for details."
                )

            files_to_clean = downloaded_volume_names.intersection(successfully_cached_volumes)
            if files_to_clean:
                logger.info(f"[Batch {batch_num}] Cleaning up {len(files_to_clean)} downloaded raw NIfTI files...")
                cleaned_count = 0
                for volume_name in files_to_clean:
                    try:
                        file_path_to_delete = get_dynamic_image_path(Path(config.paths.img_dir), volume_name, config.paths.dir_structure)
                        if file_path_to_delete.exists():
                            os.remove(file_path_to_delete)
                            cleaned_count += 1
                    except OSError as e:
                        logger.error(f"Failed to delete {volume_name}: {e}")
                logger.info(f"[Batch {batch_num}] Cleanup complete. Deleted {cleaned_count} files.")
            else:
                logger.info(f"[Batch {batch_num}] No downloaded files to clean up for this batch.")
            
            # Clean up the temporary Hugging Face cache for the current batch
            logger.info(f"[Batch {batch_num}] Cleaning up temporary Hugging Face cache: {batch_hf_cache_dir}")
            try:
                shutil.rmtree(batch_hf_cache_dir)
            except Exception as e:
                logger.error(f"[Batch {batch_num}] Failed to clean temporary Hugging Face cache {batch_hf_cache_dir}: {e}")

            write_runtime_status(
                logs_dir,
                {
                    "status": "running",
                    "stage": "batch_completed",
                    "batch": batch_num,
                    "num_batches": num_batches,
                    "cached_in_batch": len(successfully_cached_volumes),
                    "download_failed_in_batch": len(failed_downloads),
                    "workers_used_for_cache": final_workers_used,
                    **get_memory_snapshot(),
                },
            )
            flush_log_handlers()
    
    finally:
        # Final cleanup of the parent temporary directory
        if main_temp_hf_dir.exists():
            logger.info(f"Cleaning up main temporary cache directory: {main_temp_hf_dir}")
            try:
                shutil.rmtree(main_temp_hf_dir)
            except Exception as e:
                logger.error(f"Failed to clean main temporary cache directory {main_temp_hf_dir}: {e}")
        flush_log_handlers()


def cleanup_existing_raw_files(config, full_df: pd.DataFrame, cache_dir: Path):
    """
    Scans for and deletes local raw NIfTI files if a corresponding cache file exists.
    """
    logger.info("\n--- Starting Final Cleanup of Existing Raw Files ---")
    
    cleaned_count = 0
    volumes_to_check = full_df['VolumeName'].tolist()
    
    for volume_name in tqdm(volumes_to_check, desc="Cleaning up existing files"):
        raw_file_path = get_dynamic_image_path(Path(config.paths.img_dir), volume_name, config.paths.dir_structure)
        if not raw_file_path.exists():
            continue

        data_item = {"volume_name": volume_name}
        hashed_filename_str = deterministic_hash(data_item).decode('utf-8')
        cache_filepath = cache_dir / f"{hashed_filename_str}.pt"

        if cache_filepath.exists():
            try:
                os.remove(raw_file_path)
                logger.debug(f"Deleted existing raw file with cache: {raw_file_path}")
                cleaned_count += 1
            except OSError as e:
                logger.error(f"Failed to delete existing raw file {volume_name}: {e}")
    logger.info(f"Cleanup complete. Deleted {cleaned_count} raw NIfTI files that were already cached.")


def main(config_path: str, num_workers: int, batch_size: int, clean_local: bool):
    config = load_config(config_path)
    setup_logging(Path(config.paths.logs_dir))

    try:
        train_csv_path = Path(config.paths.data_dir) / config.paths.data_subsets.train
        valid_csv_path = Path(config.paths.data_dir) / config.paths.data_subsets.valid
        df_train = pd.read_csv(train_csv_path)
        df_valid = pd.read_csv(valid_csv_path)
        full_df = pd.concat([df_train, df_valid], ignore_index=True).drop_duplicates(subset=['VolumeName']).reset_index(drop=True)
        logger.info(f"Loaded a total of {len(full_df)} unique volumes for cache analysis.")

        preprocess_transforms = get_preprocessing_transforms(config)
        cache_dir = get_or_create_cache_subdirectory(Path(config.paths.cache_dir), preprocess_transforms, split="unified")
        missing_files_df = analyze_cache_state(full_df, cache_dir)
        
        num_missing = len(missing_files_df)
        logger.info("\n--- Cache Analysis Complete ---")
        logger.info(f"Total volumes required: {len(full_df)}")
        logger.info(f"Existing cached files:  {len(full_df) - num_missing}")
        logger.info(f"Missing cache files:    {num_missing}")
        
        if num_missing > 0:
            try:
                if input("\nProceed with processing missing files in batches? (Y/N): ").strip().lower() != 'y':
                    logger.info("Process declined by user.")
                else:
                    process_in_batches(config, missing_files_df, batch_size, num_workers)
            except (EOFError, KeyboardInterrupt):
                logger.info("\nProcess cancelled by user.")
        else:
            logger.info("Cache is already complete. Nothing to do for missing files.")

        if clean_local:
            cleanup_existing_raw_files(config, full_df, cache_dir)
        else:
            logger.info("Skipping final cleanup of existing raw files. Use --clean-local-raw-files to enable.")

        write_runtime_status(
            Path(config.paths.logs_dir),
            {
                "status": "completed",
                "thread": current_thread().name,
                **get_memory_snapshot(),
            },
        )
    except Exception:
        logger.exception("Fatal error while running cache generation.")
        log_memory_snapshot("fatal_error")
        write_runtime_status(
            Path(config.paths.logs_dir),
            {
                "status": "failed",
                "thread": current_thread().name,
                **get_memory_snapshot(),
            },
        )
        flush_log_handlers()
        raise

if __name__ == '__main__':
    if sys.platform == "win32":
        torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Intelligently generate a pre-processed dataset cache in batches.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers for caching.')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of files to download and cache in each batch.')
    parser.add_argument(
        '--clean-local-raw-files',
        action='store_true',
        help="If set, delete any local raw NIfTI files that already have a corresponding cache file."
    )
    args = parser.parse_args()
    
    main(args.config, args.num_workers, args.batch_size, args.clean_local_raw_files)