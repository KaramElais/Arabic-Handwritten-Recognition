import json

import matplotlib.pyplot as plt

from config import RESULTS_OUTPUT_DIR
from config import FIGURES_OUTPUT_DIR


# ============================================================
# Load Training History
# ============================================================

history_path = RESULTS_OUTPUT_DIR / "training_history.json"

with open(history_path, "r") as json_file:

    training_history = json.load(json_file)


# ============================================================
# Extract Training Data
# ============================================================

train_loss = training_history["train_loss"]

val_loss = training_history["val_loss"]

train_accuracy = training_history["train_accuracy"]

val_accuracy = training_history["val_accuracy"]

epochs = range(1, len(train_loss) + 1)


# ============================================================
# Create Figures Directory
# ============================================================

FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot Loss Curves
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_loss,
    label="Training Loss",
    linewidth=2
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss",
    linewidth=2
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

loss_figure_path = FIGURES_OUTPUT_DIR / "loss_curve.png"

plt.savefig(
    loss_figure_path,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 60)

print(f"Loss curve saved to: {loss_figure_path}")

print("=" * 60)

plt.close()


# ============================================================
# Plot Accuracy Curves
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_accuracy,
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    epochs,
    val_accuracy,
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.legend()

plt.grid(True)

accuracy_figure_path = FIGURES_OUTPUT_DIR / "accuracy_curve.png"

plt.savefig(
    accuracy_figure_path,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 60)

print(f"Accuracy curve saved to: {accuracy_figure_path}")

print("=" * 60)

plt.close()


# ============================================================
# Final Summary
# ============================================================

print("=" * 60)

print("Training visualization completed successfully.")

print(f"Figures directory: {FIGURES_OUTPUT_DIR}")

print("=" * 60)