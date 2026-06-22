"""
evaluate.py — Evaluate a saved model checkpoint on the validation set.

Usage:
    python scripts/evaluate.py --checkpoint models/best.pth

Prints per-class accuracy, overall accuracy, and a confusion matrix.
"""

import argparse

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Config (should match values used during training)
# ---------------------------------------------------------------------------
DATA_DIR = "data"
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_WORKERS = 4
# ---------------------------------------------------------------------------


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: str, num_classes: int):
    """Load the model architecture and restore weights from a checkpoint."""
    # TODO: import and instantiate the same model class used in train.py
    raise NotImplementedError("Replace this with your model definition.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate watermelon classifier")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_tf = get_val_transform()
    val_ds = ImageFolder(root=f"{DATA_DIR}/val", transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    classes = val_ds.classes
    print(f"Classes: {classes}")

    model = load_model(args.checkpoint, num_classes=len(classes)).to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
