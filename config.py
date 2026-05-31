from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DIR = PROCESSED_DATA_DIR / "train"

VAL_DIR = PROCESSED_DATA_DIR / "val"

TEST_DIR = PROCESSED_DATA_DIR / "test"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_OUTPUT_DIR = OUTPUTS_DIR / "models"

RESULTS_OUTPUT_DIR = OUTPUTS_DIR / "results"

FIGURES_OUTPUT_DIR = OUTPUTS_DIR / "figures"


# ============================================================
# Dataset Settings
# ============================================================

NUM_CLASSES = 28

RANDOM_SEED = 42


# ============================================================
# Image Settings
# ============================================================

IMAGE_SIZE = 64

CHANNELS = 1


# ============================================================
# Normalization Settings
# ============================================================

NORMALIZE_MEAN = [0.5]

NORMALIZE_STD = [0.5]


# ============================================================
# Augmentation Settings
# ============================================================

USE_AUGMENTATION = True

ROTATION_DEGREES = 10

TRANSLATE = (0.08, 0.08)

SCALE = (0.90, 1.10)

SHEAR_DEGREES = 5


# ============================================================
# DataLoader Settings
# ============================================================

BATCH_SIZE = 64

NUM_WORKERS = 0

PIN_MEMORY = False


# ============================================================
# Model Settings
# ============================================================

MODEL_NAME = "ImprovedCNN"

CONV1_OUT_CHANNELS = 32

CONV2_OUT_CHANNELS = 64

CONV3_OUT_CHANNELS = 128

FC_HIDDEN_UNITS = 256

DROPOUT_RATE = 0.5


# ============================================================
# Training Settings
# ============================================================

LEARNING_RATE = 0.001

NUM_EPOCHS = 30


# ============================================================
# Early Stopping Settings
# ============================================================

EARLY_STOPPING_PATIENCE = 5

MONITOR_METRIC = "val_loss"


# ============================================================
# Checkpoint Settings
# ============================================================

BEST_MODEL_NAME = "best_model.pth"

CHECKPOINT_NAME = "checkpoint.pth"