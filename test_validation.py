
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CLASS_NAMES, DEVICE, MODEL_PATH
from modules.model_loader import load_model


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

IMAGE_PATH = Path(
    r"D:\python\InsPLAD\InsPLAD-det\val\284-1_DJI_0495.jpg"
)

print("=" * 60)
print("POWERLINE AI - VALIDATION IMAGE TEST")
print("=" * 60)

print("Device:", DEVICE)
print("Checkpoint:", MODEL_PATH)
print("Image exists:", IMAGE_PATH.exists())

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

print("\nLoading trained Faster R-CNN model...")

model, metadata = load_model()

model.eval()

print("Model loaded successfully!")
print("Number of classes:", len(CLASS_NAMES))


# --------------------------------------------------
# 3. Load and preprocess image
# --------------------------------------------------

print("\nLoading validation image...")

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image size:", image.size)

image_array = np.array(image)

image_tensor = (
    torch.from_numpy(image_array)
    .permute(2, 0, 1)
    .float()
    .div(255.0)
    .to(DEVICE)
)

print("Image tensor shape:", tuple(image_tensor.shape))


# --------------------------------------------------
# 4. Direct model inference
# --------------------------------------------------

print("\nRunning inference...")

with torch.no_grad():
    predictions = model([image_tensor])

prediction = predictions[0]

boxes = prediction["boxes"].detach().cpu()
labels = prediction["labels"].detach().cpu()
scores = prediction["scores"].detach().cpu()

print("Inference completed!")


# --------------------------------------------------
# 5. Print prediction summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

print("Total predictions:", len(scores))

threshold = 0.10

print(f"\nPredictions with confidence >= {threshold * 100:.0f}%:")

found = False

for box, label, score in zip(boxes, labels, scores):

    confidence = score.item()
    class_id = int(label.item())

    if confidence >= threshold:

        found = True

        if 0 <= class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"Unknown class {class_id}"

        print(
            f"\nClass: {class_name}"
            f"\nClass ID: {class_id}"
            f"\nConfidence: {confidence * 100:.2f}%"
            f"\nBounding Box: {[round(x, 2) for x in box.tolist()]}"
        )

if not found:
    print("No predictions above 10% confidence.")


# --------------------------------------------------
# 6. Top 10 predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP 10 RAW PREDICTIONS")
print("=" * 60)

for index, (box, label, score) in enumerate(
    zip(boxes[:10], labels[:10], scores[:10]), start=1
):

    class_id = int(label.item())

    if 0 <= class_id < len(CLASS_NAMES):
        class_name = CLASS_NAMES[class_id]
    else:
        class_name = f"Unknown class {class_id}"

    print(
        f"{index}. {class_name} "
        f"| ID: {class_id} "
        f"| Confidence: {score.item() * 100:.2f}% "
        f"| Box: {[round(x, 2) for x in box.tolist()]}"
    )


# --------------------------------------------------
# 7. Class-wise prediction counts
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASS-WISE PREDICTION COUNTS (10% threshold)")
print("=" * 60)

class_counts = {}

for label, score in zip(labels, scores):

    if score.item() >= threshold:

        class_id = int(label.item())

        class_name = (
            CLASS_NAMES[class_id]
            if 0 <= class_id < len(CLASS_NAMES)
            else f"Unknown class {class_id}"
        )

        class_counts[class_name] = (
            class_counts.get(class_name, 0) + 1
        )

if class_counts:

    for class_name, count in sorted(
        class_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{class_name}: {count}")

else:
    print("No classes detected above threshold.")


# --------------------------------------------------
# 8. Finish
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION IMAGE TEST COMPLETED")
print("=" * 60)
