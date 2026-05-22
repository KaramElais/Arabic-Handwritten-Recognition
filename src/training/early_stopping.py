from pathlib import Path
import torch


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss
    does not improve for a given number of epochs.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
        save_path="best_model.pth"
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.save_path = Path(save_path)

        self.best_loss = None

        self.counter = 0

        self.early_stop = False

        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):

        # First epoch
        if self.best_loss is None:

            self.best_loss = val_loss

            self.best_epoch = epoch

            self.save_checkpoint(model)

            print(
                f"Initial best model saved | "
                f"Validation Loss: {val_loss:.4f}"
            )

        # Validation improved
        elif val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss

            self.best_epoch = epoch

            self.counter = 0

            self.save_checkpoint(model)

            print(
                f"Validation loss improved | "
                f"Best model saved | "
                f"Validation Loss: {val_loss:.4f}"
            )

        # No improvement
        else:

            self.counter += 1

            print(
                f"No improvement in validation loss | "
                f"Patience: {self.counter}/{self.patience}"
            )

            # Trigger early stopping
            if self.counter >= self.patience:

                self.early_stop = True

                print(
                    f"Early stopping triggered "
                    f"at epoch {epoch}"
                )

    def save_checkpoint(self, model):

        # Create directory if it does not exist
        self.save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save only model weights
        torch.save(
            model.state_dict(),
            self.save_path
        )