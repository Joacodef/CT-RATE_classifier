import argparse
import copy
import functools
import logging
import sys
from pathlib import Path

import optuna
import torch
from optuna.trial import TrialState

# Add project root to Python path to allow direct imports from src and config
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import train_model
from src.utils.logging_config import setup_logging

# Configure a basic logger for the main script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config, args: argparse.Namespace) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    This function defines the hyperparameter search space, configures, runs,
    and evaluates a single training trial. It supports staged optimization by
    using different data subsets based on the trial number.
    Args:
        trial (optuna.Trial): An Optuna trial object.
        base_config: The baseline configuration object loaded from YAML.
        args (argparse.Namespace): Command-line arguments, including trial
                                   thresholds for data subsets.
    Returns:
        float: The performance metric (max validation ROC AUC) to be maximized.
    """
    # Create a deep copy of the base config to ensure trial independence
    config = copy.deepcopy(base_config)
    trial_num = trial.number

    # --- Staged Optimization: Select data subset based on trial number ---
    base_split_dir = config.paths.labels.train.parent
    path_train_05 = base_split_dir / "train_05_percent.csv"
    path_train_20 = base_split_dir / "train_20_percent.csv"
    path_train_50 = base_split_dir / "train_50_percent.csv"
    path_train_full = config.paths.labels.train

    # Fallback to full dataset if subset files do not exist
    if not all(
        [path_train_05.exists(), path_train_20.exists(), path_train_50.exists()]
    ):
        logger.warning(
            "Training subset files not found. Using full training data for all trials."
        )
    else:
        if trial_num < args.trials_on_5_percent:
            config.paths.labels.train = path_train_05
        elif trial_num < args.trials_on_20_percent:
            config.paths.labels.train = path_train_20
        elif trial_num < args.trials_on_50_percent:
            config.paths.labels.train = path_train_50
        # else: use the full path already in config

    # --- Hyperparameter Search Space Definition ---
    config.training.learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, log=True
    )
    config.training.weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    )
    config.training.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    config.model.type = trial.suggest_categorical(
        "model_type", ["resnet3d", "densenet3d", "vit3d"]
    )
    if config.model.type == "resnet3d":
        config.model.variant = trial.suggest_categorical(
            "model_variant", ["18", "34"]
        )
    elif config.model.type == "densenet3d":
        config.model.variant = trial.suggest_categorical(
            "model_variant", ["121", "169"]
        )
    else: # vit3d
        config.model.variant = trial.suggest_categorical(
            "model_variant", ["tiny", "small", "base"]
        )

    if config.loss_function.type == "FocalLoss":
        config.loss_function.focal_loss.alpha = trial.suggest_float(
            "focal_loss_alpha", 0.1, 0.9
        )
        config.loss_function.focal_loss.gamma = trial.suggest_float(
            "focal_loss_gamma", 1.0, 5.0
        )

    # --- Trial-specific Configuration ---
    # Disable W&B logging for optimization trials
    if hasattr(config, 'wandb'):
        config.wandb.enabled = False

    trial_output_dir = config.paths.output_dir / f"trial_{trial_num}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.output_dir = trial_output_dir

    log_file = trial_output_dir / f"trial_{trial_num}.log"
    setup_logging(log_file_path=log_file)

    logger.info(
        f"Starting trial {trial_num} with training data: "
        f"'{config.paths.labels.train.name}'"
    )
    logger.info(f"Parameters: {trial.params}")

    try:
        history = train_model(config=config)
        validation_metrics = [
            m["roc_auc_macro"]
            for m in history["metrics"]
            if "roc_auc_macro" in m
        ]
        best_metric = max(validation_metrics) if validation_metrics else 0.0
        logger.info(
            f"Trial {trial_num} finished with best metric: {best_metric:.4f}"
        )
        return best_metric
    except torch.cuda.OutOfMemoryError:
        logger.error(
            f"Trial {trial_num} failed due to CUDA Out of Memory. Pruning."
        )
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        logger.error(
            f"Trial {trial_num} failed with an unexpected error: {e}",
            exc_info=True,
        )
        return 0.0


def main():
    """
    Main function to parse arguments, set up, and run the Optuna study.
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization with Optuna"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the base YAML configuration file.'
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Total number of optimization trials to run.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ct-classifier-optimization",
        help="Name for the Optuna study.",
    )
    parser.add_argument(
        "--storage-db",
        type=str,
        default="optimization_study.db",
        help="Database file for Optuna storage (e.g., 'study.db').",
    )
    # Arguments for staged optimization
    parser.add_argument(
        "--trials-on-5-percent",
        type=int,
        default=40,
        help="Run trials up to this number on 5% of the data.",
    )
    parser.add_argument(
        "--trials-on-20-percent",
        type=int,
        default=70,
        help="Run trials up to this number on 20% of the data.",
    )
    parser.add_argument(
        "--trials-on-50-percent",
        type=int,
        default=90,
        help="Run trials up to this number on 50% of the data.",
    )
    args = parser.parse_args()

    # Load the base configuration from the specified YAML file
    base_config = load_config(args.config)

    storage_name = f"sqlite:///{args.storage_db}"
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    # Pass the loaded base_config and command-line args to the objective function
    objective_with_args = functools.partial(objective, base_config=base_config, args=args)

    logger.info(
        f"Starting study '{args.study_name}' with {args.n_trials} total trials."
    )
    logger.info(f"Storage is set to: {storage_name}")
    logger.info(
        f"Staged trials: {args.trials_on_5_percent} (5%) -> "
        f"{args.trials_on_20_percent} (20%) -> "
        f"{args.trials_on_50_percent} (50%) -> "
        f"{args.n_trials} (100%)"
    )

    study.optimize(objective_with_args, n_trials=args.n_trials)

    # --- Report Study Results ---
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE]
    )

    logger.info("=" * 40)
    logger.info("Study Statistics")
    logger.info("=" * 40)
    logger.info(f"  Total Trials: {len(study.trials)}")
    logger.info(f"  Pruned Trials: {len(pruned_trials)}")
    logger.info(f"  Complete Trials: {len(complete_trials)}")
    logger.info("=" * 40)

    logger.info("Best Trial Found:")
    best_trial = study.best_trial
    logger.info(f"  Value (Max ROC AUC): {best_trial.value:.4f}")
    logger.info("  Optimal Parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    main()