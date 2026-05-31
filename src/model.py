import torch
import torch.nn as nn

from config import (
    CHANNELS,
    NUM_CLASSES,
    CONV1_OUT_CHANNELS,
    CONV2_OUT_CHANNELS,
    CONV3_OUT_CHANNELS,
    FC_HIDDEN_UNITS,
    DROPOUT_RATE,
)


class BaselineCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super(BaselineCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS, CONV1_OUT_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        flattened_features = CONV3_OUT_CHANNELS * 8 * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, FC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(FC_HIDDEN_UNITS, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


class ImprovedCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super(ImprovedCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, FC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(FC_HIDDEN_UNITS, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


def count_trainable_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    baseline_model = BaselineCNN()
    improved_model = ImprovedCNN()

    dummy_input = torch.randn(4, CHANNELS, 64, 64)

    baseline_output = baseline_model(dummy_input)
    improved_output = improved_model(dummy_input)

    print("=" * 60)
    print("CNN Models Verification")
    print("=" * 60)

    print(f"Input shape             : {dummy_input.shape}")
    print(f"Baseline output shape   : {baseline_output.shape}")
    print(f"Improved output shape   : {improved_output.shape}")

    print("-" * 60)

    print(
        f"Baseline Parameters     : "
        f"{count_trainable_parameters(baseline_model):,}"
    )

    print(
        f"Improved Parameters     : "
        f"{count_trainable_parameters(improved_model):,}"
    )

    assert baseline_output.shape == (4, NUM_CLASSES)
    assert improved_output.shape == (4, NUM_CLASSES)

    print("Both models verified successfully.")