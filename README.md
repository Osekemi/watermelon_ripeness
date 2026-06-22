# Watermelon Ripeness Classifier

An image classifier that predicts watermelon ripeness (ripe / unripe / overripe) from photos using PyTorch.

---

## Project Structure

```
watermelon_ripeness/
├── data/
│   ├── train/
│   │   ├── ripe/          # Training images — ripe watermelons
│   │   ├── unripe/        # Training images — unripe watermelons
│   │   └── overripe/      # Training images — overripe watermelons
│   └── val/
│       ├── ripe/          # Validation images — ripe watermelons
│       ├── unripe/        # Validation images — unripe watermelons
│       └── overripe/      # Validation images — overripe watermelons
├── models/                # Saved model checkpoints (.pth files)
├── scripts/
│   ├── train.py           # Train the model; saves best checkpoint to models/
│   ├── evaluate.py        # Evaluate a checkpoint on the validation set
│   └── predict.py         # Run inference on a single image
├── venv/                  # Python virtual environment (not committed)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data

Place your images in the correct subfolders under `data/`. The folder name becomes the class label — this is how PyTorch's `ImageFolder` works.

A rough split of **80% train / 20% val** per class is a good starting point.

Supported formats: `.jpg`, `.jpeg`, `.png`.

---

## Usage

All scripts are run from the **project root** (not from inside `scripts/`).

### Train

```bash
python scripts/train.py
```

Edit the config block at the top of `train.py` to change epochs, batch size, learning rate, etc. The best checkpoint is saved to `models/best.pth`.

### Evaluate

```bash
python scripts/evaluate.py --checkpoint models/best.pth
```

Prints a per-class classification report and a confusion matrix against the validation set.

### Predict (single image)

```bash
python scripts/predict.py --image path/to/your/watermelon.jpg
```

Prints the predicted class and confidence scores for all three classes.

---

## Next Steps

1. Add your model definition (e.g. fine-tune a pretrained ResNet or EfficientNet via `torchvision.models`).
2. Plug the model into the `build_model` / `load_model` stubs in each script.
3. Add data augmentation in the training transform pipeline in `train.py`.
4. Consider adding a learning-rate scheduler for longer training runs.
