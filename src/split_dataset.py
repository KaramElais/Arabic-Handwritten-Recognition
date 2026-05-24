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
# Reset Processed Directory
# =========================================================

if PROCESSED_DIR.exists():
    shutil.rmtree(PROCESSED_DIR)

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
    Collect all valid images from a folder.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Raw image folder not found: {folder}"
        )

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))

    valid_images = []

    for image_path in images:
        if is_valid_image(image_path):
            valid_images.append(image_path)

    return valid_images


def create_unique_filename(image_path, split_name, label, index):
    """
    Create a unique filename to prevent overwriting images
    when train and test folders contain duplicate names.
    """
    source_folder = image_path.parent.parent.name.replace(" ", "_")

    return (
        f"{split_name}_class_{label}_"
        f"{source_folder}_{index}_"
        f"{image_path.name}"
    )


def count_files(folder):
    """
    Count all files inside a folder recursively.
    """
    return sum(
        1 for path in folder.rglob("*")
        if path.is_file()
    )


# =========================================================
# Collect Dataset
# =========================================================

train_images = collect_images(RAW_TRAIN_DIR)

test_images = collect_images(RAW_TEST_DIR)

all_images = []

all_images.extend(train_images)
all_images.extend(test_images)

print(f"\nTotal valid images found: {len(all_images)}")


# =========================================================
# Organize Images By Class
# =========================================================

class_images = defaultdict(list)

skipped_images = []

for image_path in all_images:

    label = extract_label(image_path.name)

    if label is not None:
        class_images[label].append(image_path)

    else:
        skipped_images.append(image_path)

print(f"Number of classes found: {len(class_images)}")

if skipped_images:
    print(f"Skipped images without labels: {len(skipped_images)}")


# =========================================================
# Split Dataset
# =========================================================

statistics = []

for label, images in sorted(
    class_images.items(),
    key=lambda item: int(item[0])
):

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

        for index, image_path in enumerate(split_images):

            unique_filename = create_unique_filename(
                image_path=image_path,
                split_name=split_name,
                label=label,
                index=index
            )

            destination = class_dir / unique_filename

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
    by="class",
    key=lambda column: column.astype(int)
)

stats_path = PROCESSED_DIR / "split_statistics.csv"

stats_df.to_csv(stats_path, index=False)


# =========================================================
# Final Verification
# =========================================================

expected_train_total = stats_df["train"].sum()

expected_val_total = stats_df["val"].sum()

expected_test_total = stats_df["test"].sum()

expected_grand_total = (
    expected_train_total +
    expected_val_total +
    expected_test_total
)

actual_train_total = count_files(PROCESSED_DIR / "train")

actual_val_total = count_files(PROCESSED_DIR / "val")

actual_test_total = count_files(PROCESSED_DIR / "test")

actual_grand_total = (
    actual_train_total +
    actual_val_total +
    actual_test_total
)

print("\n" + "=" * 60)
print("DATASET SPLITTING COMPLETED")
print("=" * 60)

print("\nDataset Statistics")
print("-" * 60)

print(stats_df)

print("\nExpected Split Totals")
print("-" * 60)

print(f"Train Images     : {expected_train_total}")
print(f"Validation Images: {expected_val_total}")
print(f"Test Images      : {expected_test_total}")
print(f"Total Images     : {expected_grand_total}")

print("\nActual Files Written")
print("-" * 60)

print(f"Train Images     : {actual_train_total}")
print(f"Validation Images: {actual_val_total}")
print(f"Test Images      : {actual_test_total}")
print(f"Total Images     : {actual_grand_total}")

print("\nVerification")
print("-" * 60)

print(f"Classes Found: {len(class_images)}")

if expected_grand_total == len(all_images):
    print("Expected split count matches total valid images.")
else:
    print("WARNING: Expected split count does not match total valid images.")

if actual_grand_total == expected_grand_total:
    print("No missing images detected after copying.")
else:
    print("WARNING: Missing images detected after copying.")

if actual_train_total == expected_train_total:
    print("Train split verified.")
else:
    print("WARNING: Train split count mismatch.")

if actual_val_total == expected_val_total:
    print("Validation split verified.")
else:
    print("WARNING: Validation split count mismatch.")

if actual_test_total == expected_test_total:
    print("Test split verified.")
else:
    print("WARNING: Test split count mismatch.")

print("No preprocessing applied.")
print("Raw dataset untouched.")
print("Dataset ready for preprocessing stage.")

print("\nProcessed dataset saved to:")
print(PROCESSED_DIR)