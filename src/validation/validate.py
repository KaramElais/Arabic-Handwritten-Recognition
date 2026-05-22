import torch
from tqdm import tqdm


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Performs validation for one epoch.

    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: CUDA or CPU

    Returns:
        avg_val_loss: Average validation loss
        val_accuracy: Validation accuracy percentage
    """

    # Switch model to evaluation mode
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Disable gradient calculation
    with torch.no_grad():

        loop = tqdm(val_loader, desc="Validation", leave=False)

        for images, labels in loop:

            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Accumulate loss
            total_loss += loss.item() * images.size(0)

            # Get predicted class
            _, predicted = torch.max(outputs, dim=1)

            # Count correct predictions
            correct_predictions += (predicted == labels).sum().item()

            # Count total samples
            total_samples += labels.size(0)

    # Compute average loss
    avg_val_loss = total_loss / total_samples

    # Compute accuracy
    val_accuracy = (correct_predictions / total_samples) * 100

    return avg_val_loss, val_accuracy       