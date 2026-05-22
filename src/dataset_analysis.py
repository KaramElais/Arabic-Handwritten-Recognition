from pathlib import Path

# Dataset paths
TRAIN_PATH = Path("data/raw/ahcd/Train Images 13440x32x32/train")
TEST_PATH = Path("data/raw/ahcd/Test Images 3360x32x32/test")

print("Train path exists:", TRAIN_PATH.exists())
print("Test path exists:", TEST_PATH.exists())

# Count images
train_images = list(TRAIN_PATH.glob("*.png"))
test_images = list(TEST_PATH.glob("*.png"))

print(f"Train Images: {len(train_images)}")
print(f"Test Images: {len(test_images)}")
print(f"Total Images: {len(train_images) + len(test_images)}")  

# =========================
# Extract labels from filenames
# =========================

def extract_label(image_path):
    file_name = image_path.stem
    label = file_name.split("_label_")[-1]
    return int(label)


train_labels = [extract_label(img) for img in train_images]
test_labels = [extract_label(img) for img in test_images]

print("\nClass Statistics")
print("-" * 40)
print(f"Number of train classes: {len(set(train_labels))}")
print(f"Number of test classes: {len(set(test_labels))}")

print("\nTrain labels:")
print(sorted(set(train_labels)))

print("\nTest labels:")
print(sorted(set(test_labels)))

from collections import Counter

# =========================
# Class Distribution Analysis
# =========================

train_class_counts = Counter(train_labels)
test_class_counts = Counter(test_labels)

print("\nTrain Class Distribution")
print("-" * 40)

for label, count in sorted(train_class_counts.items()):
    print(f"Class {label}: {count} images")

print("\nTest Class Distribution")
print("-" * 40)

for label, count in sorted(test_class_counts.items()):
    print(f"Class {label}: {count} images")

    # =========================
# Image Properties Analysis
# =========================

from PIL import Image

image_properties = []

all_images = train_images + test_images

for img_path in all_images:
    with Image.open(img_path) as img:
        image_properties.append({
            "file_name": img_path.name,
            "split": "train" if img_path in train_images else "test",
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format
        })

print("\nImage Properties Analysis")
print("-" * 40)

widths = [item["width"] for item in image_properties]
heights = [item["height"] for item in image_properties]
modes = [item["mode"] for item in image_properties]
formats = [item["format"] for item in image_properties]

print(f"Unique image dimensions: {sorted(set(zip(widths, heights)))}")
print(f"Image modes: {set(modes)}")
print(f"Image formats: {set(formats)}")

# =========================
# Corrupted Image Check
# =========================

corrupted_images = []

for img_path in all_images:
    try:
        with Image.open(img_path) as img:
            img.verify()
    except Exception:
        corrupted_images.append(str(img_path))

print("\nCorrupted Image Check")
print("-" * 40)
print(f"Number of corrupted images: {len(corrupted_images)}")

if corrupted_images:
    print("Corrupted images:")
    for img in corrupted_images:
        print(img)

        # =========================
# Duplicate Image Check
# =========================

import hashlib

hashes = {}
duplicate_images = []

for img_path in all_images:
    file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()

    if file_hash in hashes:
        duplicate_images.append((str(img_path), str(hashes[file_hash])))
    else:
        hashes[file_hash] = img_path

print("\nDuplicate Image Check")
print("-" * 40)
print(f"Number of duplicate images: {len(duplicate_images)}")

if duplicate_images:
    print("Duplicate image pairs:")
    for dup, original in duplicate_images[:20]:
        print(f"{dup} == {original}")

        # =========================
# Save Class Distribution Plot
# =========================

import matplotlib.pyplot as plt

OUTPUT_FIGURES_PATH = Path("outputs/figures")
OUTPUT_FIGURES_PATH.mkdir(parents=True, exist_ok=True)

labels = sorted(train_class_counts.keys())
train_counts = [train_class_counts[label] for label in labels]
test_counts = [test_class_counts[label] for label in labels]

plt.figure(figsize=(14, 6))
plt.bar(labels, train_counts)
plt.title("Training Set Class Distribution")
plt.xlabel("Class Label")
plt.ylabel("Number of Images")
plt.xticks(labels)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES_PATH / "train_class_distribution.png")
plt.close()

plt.figure(figsize=(14, 6))
plt.bar(labels, test_counts)
plt.title("Test Set Class Distribution")
plt.xlabel("Class Label")
plt.ylabel("Number of Images")
plt.xticks(labels)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES_PATH / "test_class_distribution.png")
plt.close()

print("\nPlots Saved")
print("-" * 40)
print("Saved: outputs/figures/train_class_distribution.png")
print("Saved: outputs/figures/test_class_distribution.png")

# =========================
# Save Random Sample Images
# =========================

import random

sample_images = random.sample(all_images, 16)

plt.figure(figsize=(8, 8))

for_index = 1
for img_path in sample_images:
    label = extract_label(img_path)
    image = Image.open(img_path)

    plt.subplot(4, 4, for_index)
    plt.imshow(image, cmap="gray")
    plt.title(f"Class {label}")
    plt.axis("off")

    for_index += 1

plt.tight_layout()
plt.savefig(OUTPUT_FIGURES_PATH / "random_sample_images.png")
plt.close()

print("Saved: outputs/figures/random_sample_images.png")

# =========================
# Save Dataset Summary CSV
# =========================

import pandas as pd

summary_data = {
    "Metric": [
        "Total Images",
        "Train Images",
        "Test Images",
        "Number of Classes",
        "Image Dimensions",
        "Image Mode",
        "Image Format",
        "Corrupted Images",
        "Duplicate Images"
    ],
    "Value": [
        len(all_images),
        len(train_images),
        len(test_images),
        len(set(train_labels)),
        "32x32",
        "Grayscale (L)",
        "PNG",
        len(corrupted_images),
        len(duplicate_images)
    ]
}

summary_df = pd.DataFrame(summary_data)

OUTPUT_REPORTS_PATH = Path("outputs/reports")
OUTPUT_REPORTS_PATH.mkdir(parents=True, exist_ok=True)

summary_df.to_csv(
    OUTPUT_REPORTS_PATH / "dataset_summary.csv",
    index=False
)

print("\nDataset Summary Saved")
print("-" * 40)
print("Saved: outputs/reports/dataset_summary.csv")