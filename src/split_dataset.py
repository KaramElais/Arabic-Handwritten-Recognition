from pathlib import Path
import shutil
import random
import re
from collections import defaultdict
import pandas as pd
from PIL import Image


# =========================================================
# Configuration
# =========================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)

RAW_TRAIN_DIR = Path(
    "data/raw/ahcd/Train Images 13440x32x32/train"
)

RAW_TEST_DIR = Path(
    "data/raw/ahcd/Test Images 3360x32x32/test"
)

PROCESSED_DIR = Path("data/processed")

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


# =========================================================
# Create Processed Folders
# =========================================================

for split in ["train", "val", "test"]:
    (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)


# =========================================================
# Helper Functions
# =========================================================

def extract_label(filename):
    """
    Extract class label from filename.

    Example:
    id_10000_label_18.png -> 18
    """
    match = re.search(r"label_(\d+)", filename)

    if match:
        return match.group(1)

    return None


def is_valid_image(image_path):
    """
    Verify image integrity.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images(folder):
    """
    Collect all valid images from folder.
    """
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))

    valid_images = []

    for image_path in images:
        if is_valid_image(image_path):
            valid_images.append(image_path)

    return valid_images


# =========================================================
# Collect Dataset
# =========================================================

all_images = []

train_images = collect_images(RAW_TRAIN_DIR)
test_images = collect_images(RAW_TEST_DIR)

all_images.extend(train_images)
all_images.extend(test_images)

print(f"\nTotal valid images found: {len(all_images)}")


# =========================================================
# Organize Images By Class
# =========================================================

class_images = defaultdict(list)

for image_path in all_images:

    label = extract_label(image_path.name)

    if label is not None:
        class_images[label].append(image_path)

print(f"Number of classes found: {len(class_images)}")


# =========================================================
# Split Dataset
# =========================================================

statistics = []

for label, images in sorted(class_images.items()):

    random.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)

    train_split = images[:train_count]

    val_split = images[
        train_count:train_count + val_count
    ]

    test_split = images[
        train_count + val_count:
    ]

    split_map = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    for split_name, split_images in split_map.items():

        class_dir = PROCESSED_DIR / split_name / label
        class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:

            destination = class_dir / image_path.name

            shutil.copy2(image_path, destination)

    statistics.append({
        "class": label,
        "total": total_images,
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split)
    })


# =========================================================
# Create Statistics DataFrame
# =========================================================

stats_df = pd.DataFrame(statistics)

stats_df = stats_df.sort_values(
    by="class"
)

stats_path = PROCESSED_DIR / "split_statistics.csv"

stats_df.to_csv(stats_path, index=False)


# =========================================================
# Final Verification
# =========================================================

train_total = stats_df["train"].sum()
val_total = stats_df["val"].sum()
test_total = stats_df["test"].sum()

grand_total = (
    train_total +
    val_total +
    test_total
)

print("\n" + "=" * 60)
print("DATASET SPLITTING COMPLETED")
print("=" * 60)

print("\nDataset Statistics")
print("-" * 60)

print(stats_df)

print("\nSplit Totals")
print("-" * 60)

print(f"Train Images     : {train_total}")
print(f"Validation Images: {val_total}")
print(f"Test Images      : {test_total}")
print(f"Total Images     : {grand_total}")

print("\nVerification")
print("-" * 60)

print(f"Classes Found: {len(class_images)}")

if grand_total == len(all_images):
    print("No missing images detected.")
else:
    print("WARNING: Missing images detected.")

print("No preprocessing applied.")
print("Raw dataset untouched.")
print("Dataset ready for preprocessing stage.")

print("\nProcessed dataset saved to:")
print(PROCESSED_DIR)