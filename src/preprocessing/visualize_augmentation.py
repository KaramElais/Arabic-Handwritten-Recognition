import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.transforms import get_train_transform


def denormalize(tensor):
    return tensor * 0.5 + 0.5


def visualize_augmentations(image_path, num_samples=8):

    image = Image.open(image_path).convert("L")

    transform = get_train_transform(
        use_augmentation=True
    )

    plt.figure(figsize=(12, 4))

    for i in range(num_samples):

        augmented = transform(image)

        augmented = denormalize(augmented)

        plt.subplot(2, 4, i + 1)

        plt.imshow(
            augmented.squeeze(),
            cmap="gray"
        )

        plt.axis("off")

        plt.title(f"Aug {i+1}")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    sample_image = (
        "data/processed/train/20/"
        "id_12253_label_20.png"
    )

    visualize_augmentations(sample_image)