from pathlib import Path
import pandas as pd
import json
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path("outputs/results")

EXPERIMENT_LOG_CSV = RESULTS_DIR / "experiment_log.csv"
EXPERIMENT_LOG_JSON = RESULTS_DIR / "experiment_log.json"
FINAL_METRICS_SUMMARY_CSV = RESULTS_DIR / "final_metrics_summary.csv"


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# EXPERIMENT LOG STRUCTURE
# ============================================================

EXPERIMENT_COLUMNS = [
    # Identification
    "experiment_id",
    "experiment_name",
    "experiment_date",

    # Model information
    "model_name",
    "dataset_version",

    # Training configuration
    "image_size",
    "batch_size",
    "learning_rate",
    "epochs",
    "best_epoch",

    # Optimization
    "optimizer",
    "loss_function",

    # Data and augmentation
    "augmentation_used",
    "augmentation_details",

    # Training control
    "early_stopping_patience",

    # Hardware
    "device",
    "gpu_name",
    "kaggle_gpu_used",

    # Performance metrics
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "test_loss",
    "precision",
    "recall",
    "f1_score",

    # Notes
    "notes"
]

# ============================================================
# INITIALIZE LOG FILES
# ============================================================

def initialize_experiment_logs():
    """
    Create experiment tracking files if they do not exist.
    Existing files will never be overwritten.
    """

    # CSV Log
    if not EXPERIMENT_LOG_CSV.exists():

        empty_df = pd.DataFrame(columns=EXPERIMENT_COLUMNS)

        empty_df.to_csv(
            EXPERIMENT_LOG_CSV,
            index=False
        )

        print(f"Created: {EXPERIMENT_LOG_CSV}")

    # JSON Log
    if not EXPERIMENT_LOG_JSON.exists():

        with open(EXPERIMENT_LOG_JSON, "w") as f:
            json.dump([], f, indent=4)

        print(f"Created: {EXPERIMENT_LOG_JSON}")

    # Final Summary CSV
    if not FINAL_METRICS_SUMMARY_CSV.exists():

        summary_columns = [
            "experiment_id",
            "experiment_name",
            "model_name",
            "test_accuracy",
            "test_loss",
            "precision",
            "recall",
            "f1_score"
        ]

        summary_df = pd.DataFrame(columns=summary_columns)

        summary_df.to_csv(
            FINAL_METRICS_SUMMARY_CSV,
            index=False
        )

        print(f"Created: {FINAL_METRICS_SUMMARY_CSV}")


# ============================================================
# UNIQUE EXPERIMENT ID GENERATOR
# ============================================================

def generate_experiment_id():
    """
    Generate a unique experiment ID based on current date and log count.

    Example:
    EXP_20260529_001
    """

    today = datetime.now().strftime("%Y%m%d")

    if EXPERIMENT_LOG_CSV.exists():
        existing_logs = pd.read_csv(EXPERIMENT_LOG_CSV)
        next_number = len(existing_logs) + 1
    else:
        next_number = 1

    experiment_id = f"EXP_{today}_{next_number:03d}"

    return experiment_id


# ============================================================
# SAVE EXPERIMENT RECORD
# ============================================================

def save_experiment_record(experiment_data):
    """
    Save experiment results safely to:
    - CSV log
    - JSON log
    - Final metrics summary
    """

    # --------------------------
    # CSV LOG
    # --------------------------

    csv_df = pd.read_csv(EXPERIMENT_LOG_CSV)

    if experiment_data["experiment_id"] in csv_df.get("experiment_id", []).values:
        print(
            f"Experiment ID already exists: "
            f"{experiment_data['experiment_id']}"
        )
        return

    new_row_df = pd.DataFrame([experiment_data])

    csv_df = pd.concat(
        [csv_df, new_row_df],
        ignore_index=True
    )

    csv_df.to_csv(
        EXPERIMENT_LOG_CSV,
        index=False
    )

    # --------------------------
    # JSON LOG
    # --------------------------

    with open(EXPERIMENT_LOG_JSON, "r") as f:
        json_data = json.load(f)

    json_data.append(experiment_data)

    with open(EXPERIMENT_LOG_JSON, "w") as f:
        json.dump(
            json_data,
            f,
            indent=4
        )

    # --------------------------
    # FINAL SUMMARY
    # --------------------------

    summary_row = {
        "experiment_id": experiment_data["experiment_id"],
        "experiment_name": experiment_data["experiment_name"],
        "model_name": experiment_data["model_name"],
        "test_accuracy": experiment_data["test_accuracy"],
        "test_loss": experiment_data["test_loss"],
        "precision": experiment_data["precision"],
        "recall": experiment_data["recall"],
        "f1_score": experiment_data["f1_score"]
    }

    summary_df = pd.read_csv(FINAL_METRICS_SUMMARY_CSV)

    summary_row_df = pd.DataFrame([summary_row])

    summary_df = pd.concat(
        [summary_df, summary_row_df],
        ignore_index=True
    )

    summary_df.to_csv(
        FINAL_METRICS_SUMMARY_CSV,
        index=False
    )

    print(
        f"Experiment saved successfully: "
        f"{experiment_data['experiment_id']}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    initialize_experiment_logs()

    new_experiment_id = generate_experiment_id()
    print(f"Generated experiment ID: {new_experiment_id}")

 