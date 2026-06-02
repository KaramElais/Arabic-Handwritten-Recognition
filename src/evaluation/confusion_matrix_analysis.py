import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from src.model import ImprovedCNN
from src.dataloader import create_dataloaders


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Final Confusion Matrix and Misclassification Analysis")
    print("=" * 60)
    print(f"Using device: {device}")

    outputs_dir = Path("outputs")
    figures_dir = outputs_dir / "figures"
    results_dir = outputs_dir / "results"
    models_dir = outputs_dir / "models"

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "ImprovedCNN_best_model.pth"

    if not model_path.exists():
        model_path = outputs_dir / "final_package" / "model" / "final_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            "Final model checkpoint not found. Expected ImprovedCNN_best_model.pth "
            "or outputs/final_package/model/final_model.pth"
        )

    print(f"Loading model from: {model_path}")

    model = ImprovedCNN().to(device)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )
    model.eval()

    _, _, test_loader = create_dataloaders()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    all_confidences = []
    all_is_correct = []

    test_loss = 0.0
    total = 0
    correct = 0

    print("Running inference on test set...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(
                probabilities,
                dim=1
            )

            test_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_is_correct.extend(
                (predictions == labels).cpu().numpy()
            )

    test_loss = test_loss / total
    test_accuracy = 100 * correct / total

    print("\nFinal Test Results")
    print("-" * 60)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.2f}%")

    class_names = [str(i) for i in range(1, 29)]

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # ==========================
    # Raw Confusion Matrix
    # ==========================

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest")
    plt.title("ImprovedCNN Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    raw_cm_path = figures_dir / "improvedcnn_confusion_matrix.png"
    plt.savefig(raw_cm_path, bbox_inches="tight", dpi=300)
    plt.close()

    # ==========================
    # Normalized Confusion Matrix
    # ==========================

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_normalized, interpolation="nearest")
    plt.title("ImprovedCNN Normalized Confusion Matrix")
    plt.colorbar()

    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    normalized_cm_path = figures_dir / "improvedcnn_normalized_confusion_matrix.png"
    plt.savefig(normalized_cm_path, bbox_inches="tight", dpi=300)
    plt.close()

    # ==========================
    # Classification Report
    # ==========================

    report_text = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )

    report_path = results_dir / "ImprovedCNN_classification_report.txt"

    with open(report_path, "w") as file:
        file.write(report_text)

    print("\nClassification Report")
    print("-" * 60)
    print(report_text)

    # ==========================
    # Misclassification Analysis
    # ==========================

    misclassified_rows = []

    for index, (
        true_label,
        predicted_label,
        confidence,
        is_correct
    ) in enumerate(zip(
        all_labels,
        all_predictions,
        all_confidences,
        all_is_correct
    )):
        if not is_correct:
            misclassified_rows.append({
                "sample_index": index,
                "true_class": int(true_label) + 1,
                "predicted_class": int(predicted_label) + 1,
                "confidence": float(confidence)
            })

    misclassified_df = pd.DataFrame(misclassified_rows)

    misclassified_path = results_dir / "ImprovedCNN_misclassifications.csv"
    misclassified_df.to_csv(
        misclassified_path,
        index=False
    )

    summary = {
        "model_name": "ImprovedCNN",
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "total_test_samples": total,
        "correct_predictions": correct,
        "incorrect_predictions": len(misclassified_df),
        "confusion_matrix_path": str(raw_cm_path),
        "normalized_confusion_matrix_path": str(normalized_cm_path),
        "classification_report_path": str(report_path),
        "misclassification_csv_path": str(misclassified_path)
    }

    summary_path = results_dir / "ImprovedCNN_confusion_analysis_summary.json"

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=4)

    print("\nSaved Outputs")
    print("-" * 60)
    print(raw_cm_path)
    print(normalized_cm_path)
    print(report_path)
    print(misclassified_path)
    print(summary_path)


if __name__ == "__main__":
    main()