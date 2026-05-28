import os
import sys

sys.path.append("/kaggle/working/Arabic-Handwritten-Recognition")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F

from PIL import Image

from config import *
from src.model import ArabicCNN
from src.dataloader import get_test_loader


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("/kaggle/working/outputs/figures", exist_ok=True)
os.makedirs("/kaggle/working/outputs/results", exist_ok=True)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CONFUSION MATRIX & MISCLASSIFICATION ANALYSIS")
print("=" * 60)
print(f"Using device: {device}")


# ============================================================
# LOAD TEST DATALOADER
# ============================================================

test_loader = get_test_loader()

print("\nTest loader loaded successfully")


# ============================================================
# LOAD MODEL
# ============================================================

model = ArabicCNN().to(device)

model_path = "/kaggle/working/outputs/models/best_model.pth"

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Best model loaded successfully")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [str(i) for i in range(1, NUM_CLASSES + 1)]

print("\nClass Names:")
print(class_names)


# ============================================================
# STORE RESULTS
# ============================================================

all_labels = []
all_predictions = []
all_confidences = []

misclassified_images = []
misclassified_true = []
misclassified_pred = []
misclassified_conf = []


# ============================================================
# INFERENCE LOOP
# ============================================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = F.softmax(outputs, dim=1)

        confidences, predictions = torch.max(probabilities, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

        # Store misclassified samples
        for i in range(len(labels)):

            if predictions[i] != labels[i]:

                misclassified_images.append(images[i].cpu())
                misclassified_true.append(labels[i].cpu().item())
                misclassified_pred.append(predictions[i].cpu().item())
                misclassified_conf.append(confidences[i].cpu().item())


print("\nInference completed successfully")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(all_labels, all_predictions)

print("\nConfusion Matrix Shape:")
print(cm.shape)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


# ============================================================
# SAVE RAW CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "/kaggle/working/outputs/figures/confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nRaw confusion matrix saved")


# ============================================================
# SAVE NORMALIZED CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "/kaggle/working/outputs/figures/confusion_matrix_normalized.png",
    dpi=300
)

plt.close()

print("Normalized confusion matrix saved")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    "/kaggle/working/outputs/results/classification_report.csv",
    index=True
)

print("\nClassification report saved")


# ============================================================
# SAVE MISCLASSIFICATION CSV
# ============================================================

misclassification_df = pd.DataFrame({
    "true_label": misclassified_true,
    "predicted_label": misclassified_pred,
    "confidence": misclassified_conf
})

misclassification_df.to_csv(
    "/kaggle/working/outputs/results/misclassification_analysis.csv",
    index=False
)

print("Misclassification CSV saved")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print(f"Total test samples       : {len(all_labels)}")
print(f"Correct predictions      : {np.sum(np.array(all_labels) == np.array(all_predictions))}")
print(f"Incorrect predictions    : {len(misclassified_true)}")

print("\nStep 13 analysis completed successfully")