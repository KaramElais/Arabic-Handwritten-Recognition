from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import config

from src.preprocessing.transforms import (
    get_train_transform,
    get_eval_transform
)


class NaturalSortedImageFolder(ImageFolder):
    """
    ImageFolder with numeric class sorting.

    This ensures folders named:
    1, 2, 3, ..., 28

    are ordered numerically instead of alphabetically.
    """

    def find_classes(self, directory):
        classes = [
            entry.name
            for entry in directory.iterdir()
            if entry.is_dir()
        ]

        classes = sorted(classes, key=lambda x: int(x))

        class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(classes)
        }

        return classes, class_to_idx


def create_datasets():
    """
    Create train, validation, and test datasets.
    """

    train_dataset = NaturalSortedImageFolder(
        root=config.TRAIN_DIR,
        transform=get_train_transform(
            use_augmentation=config.USE_AUGMENTATION
        )
    )

    val_dataset = NaturalSortedImageFolder(
        root=config.VAL_DIR,
        transform=get_eval_transform()
    )

    test_dataset = NaturalSortedImageFolder(
        root=config.TEST_DIR,
        transform=get_eval_transform()
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders():
    """
    Create DataLoaders for train, validation, and test datasets.
    """

    train_dataset, val_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader