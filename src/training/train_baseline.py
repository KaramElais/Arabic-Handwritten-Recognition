import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_OUTPUT_DIR,
    BEST_MODEL_NAME,
)

from src.model import BaselineCNN
from src.dataloader import create_dataloaders
from src.validation.validate import validate_one_epoch
from src.training.early_stopping import EarlyStopping


# ============================================================
# Device Setup
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available. Training must be run on Kaggle GPU only."
    )

device = torch.device("cuda")

print("=" * 60)
print("Device Configuration")
print("=" * 60)
print(f"Using device : {device}")
print(f"GPU Name     : {torch.cuda.get_device_name(0)}")
print("=" * 60)


# ============================================================
# DataLoader Setup
# ============================================================

train_loader, val_loader, test_loader = create_dataloaders()


# ============================================================
# Model Setup
# ============================================================

model = BaselineCNN().to(device)


# ============================================================
# Loss Function and Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE,
    save_path=MODEL_OUTPUT_DIR / BEST_MODEL_NAME
)

print("Early stopping initialized.")
print(f"Best model will be saved to: {MODEL_OUTPUT_DIR / BEST_MODEL_NAME}")

# ============================================================
# Training History
# ============================================================

training_history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}
best_train_accuracy = 0.0

# ============================================================
# Training Information
# ============================================================

print("=" * 60)
print("Baseline CNN Training Started")
print("=" * 60)
print(f"Epochs        : {NUM_EPOCHS}")
print(f"Device        : {device}")
print("=" * 60)

# ============================================================
# Training Loop
# ============================================================

for epoch in range(NUM_EPOCHS):
    model.train()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    for images, labels in progress_bar:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        correct_predictions += (
            predicted == labels
        ).sum().item()

        total_samples += labels.size(0)

        current_accuracy = (
            100 * correct_predictions / total_samples
        )

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Accuracy": f"{current_accuracy:.2f}%"
        })

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = (
        100 * correct_predictions / total_samples
    )

    if epoch_accuracy > best_train_accuracy:
       best_train_accuracy = epoch_accuracy

    # ============================================================
    # Validation Phase
    # ============================================================

    val_loss, val_accuracy = validate_one_epoch(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )

    training_history["train_loss"].append(epoch_loss)

    training_history["train_accuracy"].append(epoch_accuracy)

    training_history["val_loss"].append(val_loss)

    training_history["val_accuracy"].append(val_accuracy)

    early_stopping(
        val_loss=val_loss,
        model=model,
        epoch=epoch + 1
    )

    if early_stopping.early_stop:

        print("\nEarly stopping activated.")

        print(
            f"Best model was saved at epoch "
            f"{early_stopping.best_epoch}"
    )

        break
 


    print("-" * 60)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Summary"
    )

    print(
        f"Training Loss    : {epoch_loss:.4f}"
    )

    print(
        f"Training Accuracy: {epoch_accuracy:.2f}%"
    )

    print(
        f"Validation Loss  : {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    print("-" * 60)
    print()

    # ============================================================
    # Training Behavior Analysis
    # ============================================================

    if epoch_accuracy - val_accuracy > 10:
        print("Warning: Possible overfitting detected.")

    elif epoch_accuracy < 60 and val_accuracy < 60:
        print("Warning: Possible underfitting detected.")

    elif val_loss > epoch_loss:
        print("Notice: Validation loss is higher than training loss.")

    else:
        print("Training and validation behavior looks stable.")


print(
    f"\nBest Training Accuracy: "
    f"{best_train_accuracy:.2f}%"
)

if early_stopping.early_stop:

    print("Training stopped before reaching maximum epochs.")

else:

    print("Training completed all epochs.")

print(f"Best validation loss: {early_stopping.best_loss:.4f}")

print(f"Best epoch: {early_stopping.best_epoch}")

print(
    f"Best model saved at: "
    f"{MODEL_OUTPUT_DIR / BEST_MODEL_NAME}"
)