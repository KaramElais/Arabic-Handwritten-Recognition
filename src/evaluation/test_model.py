import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from config import (
    BATCH_SIZE,
    MODEL_OUTPUT_DIR,
    BEST_MODEL_NAME
)

from src.model import BaselineCNN

from src.dataloader import create_dataloaders


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Final Model Evaluation Started")
print("=" * 60)

print(f"Using device: {DEVICE}")

# =========================================================
# Load Test Data
# =========================================================

_, _, test_loader = create_dataloaders()

print("\nTest Data Loaded Successfully")

# =========================================================
# Load Best Saved Model
# =========================================================

model = BaselineCNN().to(DEVICE)

checkpoint_path = MODEL_OUTPUT_DIR / BEST_MODEL_NAME

print(f"\nLoading best model from: {checkpoint_path}")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=DEVICE
    )
)

model.eval()

print("Best model loaded successfully")

# =========================================================
# Final Testing Loop
# =========================================================

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct_predictions = 0
total_samples = 0

all_true_labels = []
all_predicted_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        correct_predictions += (
            predicted == labels
        ).sum().item()

        total_samples += labels.size(0)

        all_true_labels.extend(
            labels.cpu().numpy()
        )

        all_predicted_labels.extend(
            predicted.cpu().numpy()
        )

        # =========================================================
# Calculate Final Metrics
# =========================================================

average_test_loss = test_loss / total_samples

test_accuracy = (
    correct_predictions / total_samples
) * 100

precision = precision_score(
    all_true_labels,
    all_predicted_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_true_labels,
    all_predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_true_labels,
    all_predicted_labels,
    average="weighted",
    zero_division=0
)


# =========================================================
# Print Final Evaluation Results
# =========================================================

print("\n" + "=" * 60)
print("FINAL TESTING RESULTS")
print("=" * 60)

print(f"Test Loss     : {average_test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.2f}%")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-score      : {f1:.4f}")

print("\nClassification Report")
print("-" * 60)

print(
    classification_report(
        all_true_labels,
        all_predicted_labels,
        zero_division=0
    )
)