from pathlib import Path
import sys

# =========================================================
# Fix Python Import Path
# =========================================================
# This allows the script to import config.py from project root
# even when this file is inside src/preprocessing/

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Imports
# =========================================================

from PIL import Image

from config import TRAIN_DIR
from src.preprocessing.transforms import get_train_transform


# =========================================================
# Preprocessing Verification
# =========================================================

def main():
    """
    Verify that one image can pass through the preprocessing pipeline.
    This does NOT create DataLoaders.
    This does NOT train any model.
    This does NOT modify any image files.
    """

    transform = get_train_transform()

    train_dir = Path(TRAIN_DIR)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    sample_images = list(train_dir.rglob("*.png"))

    if len(sample_images) == 0:
        raise FileNotFoundError("No PNG images found inside the training directory.")

    sample_image_path = sample_images[0]

    image = Image.open(sample_image_path)

    processed_image = transform(image)

    print("\n" + "=" * 60)
    print("PREPROCESSING VERIFICATION")
    print("=" * 60)

    print(f"Sample image path     : {sample_image_path}")
    print(f"Original image mode   : {image.mode}")
    print(f"Processed tensor shape: {processed_image.shape}")
    print(f"Tensor min value      : {processed_image.min().item():.4f}")
    print(f"Tensor max value      : {processed_image.max().item():.4f}")

    print("\nExpected tensor shape : torch.Size([1, 64, 64])")
    print("No image files were modified.")
    print("No augmentation was applied.")
    print("Preprocessing verification completed successfully.")


if __name__ == "__main__":
    main()