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

        # ====================================================
        # Feature Extraction Part
        # ====================================================

        self.features = nn.Sequential(

            # ---------------- Block 1 ----------------

            nn.Conv2d(
                in_channels=CHANNELS,
                out_channels=CONV1_OUT_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # ---------------- Block 2 ----------------

            nn.Conv2d(
                in_channels=CONV1_OUT_CHANNELS,
                out_channels=CONV2_OUT_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # ---------------- Block 3 ----------------

            nn.Conv2d(
                in_channels=CONV2_OUT_CHANNELS,
                out_channels=CONV3_OUT_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # ====================================================
        # Classification Part
        # ====================================================

        # After 3 MaxPooling operations:
        # 64 -> 32 -> 16 -> 8

        flattened_features = CONV3_OUT_CHANNELS * 8 * 8

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                in_features=flattened_features,
                out_features=FC_HIDDEN_UNITS
            ),

            nn.ReLU(),

            nn.Dropout(
                p=DROPOUT_RATE
            ),

            nn.Linear(
                in_features=FC_HIDDEN_UNITS,
                out_features=num_classes
            )
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

    model = BaselineCNN()

    dummy_input = torch.randn(4, CHANNELS, 64, 64)

    output = model(dummy_input)

    print("=" * 60)
    print("Baseline CNN Verification")
    print("=" * 60)

    print(f"Input shape  : {dummy_input.shape}")

    print(f"Output shape : {output.shape}")

    print(
        f"Trainable Parameters: "
        f"{count_trainable_parameters(model):,}"
    )

    assert output.shape == (4, NUM_CLASSES)

    print("Model verification completed successfully.")