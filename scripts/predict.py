"""
predict.py — Run inference on a single image.

Usage:
    python scripts/predict.py --image path/to/watermelon.jpg \
                              --checkpoint models/best.pth

Prints the predicted class and confidence scores for all classes.
"""

import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

# Class names must match the subfolder names under data/train/
CLASSES = ["overripe", "ripe", "unripe"]  # alphabetical — matches ImageFolder order

IMG_SIZE = 224


def get_inference_transform():
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


def predict(image_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, num_classes=len(CLASSES)).to(device)
    model.eval()

    tf = get_inference_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = tf(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    predicted_class = CLASSES[probs.index(max(probs))]

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class.upper()}")
    print("\nConfidence scores:")
    for cls, prob in zip(CLASSES, probs):
        bar = "#" * int(prob * 30)
        print(f"  {cls:<10} {prob:.2%}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Predict watermelon ripeness")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--checkpoint", default="models/best.pth",
                        help="Path to .pth checkpoint (default: models/best.pth)")
    args = parser.parse_args()

    predict(args.image, args.checkpoint)


if __name__ == "__main__":
    main()
