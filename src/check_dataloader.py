import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from src.dataloader import (
    create_datasets,
    create_dataloaders
)


def check_datasets():

    train_dataset, val_dataset, test_dataset = create_datasets()

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    print(f"Train samples: {len(train_dataset)}")

    print(f"Validation samples: {len(val_dataset)}")

    print(f"Test samples: {len(test_dataset)}")

    print("\nClass mapping:")

    print(train_dataset.class_to_idx)

    print("\nNumber of classes:")

    print(len(train_dataset.classes))

    print("\nClass names:")

    print(train_dataset.classes)


def check_dataloaders():

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    images, labels = next(iter(train_loader))

    print("\n" + "=" * 60)

    print("DATALOADER VERIFICATION")

    print("=" * 60)

    print(f"Batch image shape: {images.shape}")

    print(f"Batch label shape: {labels.shape}")

    print(
        f"Image tensor min: "
        f"{images.min().item():.4f}"
    )

    print(
        f"Image tensor max: "
        f"{images.max().item():.4f}"
    )

    print(f"Labels: {labels}")

    print("\nExpected image shape:")

    print("[BATCH_SIZE, 1, 64, 64]")


if __name__ == "__main__":

    check_datasets()

    check_dataloaders()