"""
train.py — Train the watermelon ripeness classifier.

Usage:
    python scripts/train.py

Expects data laid out as:
    data/train/<class>/image.jpg
    data/val/<class>/image.jpg

Saves the best checkpoint to models/best.pth.
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMG_SIZE = 224
NUM_WORKERS = 4
# ---------------------------------------------------------------------------


def get_transforms():
    """Return train and validation image transform pipelines."""
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # TODO: add data augmentation here (RandomHorizontalFlip, ColorJitter, etc.)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_dataloaders(train_tf, val_tf):
    """Load train and val datasets from DATA_DIR using ImageFolder."""
    train_ds = ImageFolder(root=f"{DATA_DIR}/train", transform=train_tf)
    val_ds = ImageFolder(root=f"{DATA_DIR}/val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int):
    """Instantiate and return the classifier model."""
    # TODO: define or import your model here
    raise NotImplementedError("Replace this with your model definition.")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch; return average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Run evaluation; return average loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()
    train_loader, val_loader, classes = build_dataloaders(train_tf, val_tf)
    print(f"Classes: {classes}")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_DIR}/best.pth")
            print(f"  Saved new best model (val acc {val_acc:.4f})")

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
