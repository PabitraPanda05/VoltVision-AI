
import json
import csv
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from config import DEVICE
from modules.model_loader import load_model
from modules.detection import detect


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

PROJECT_DIR = Path(r"D:\python\PowerLine_AI")
DATASET_DIR = Path(r"D:\python\InsPLAD\InsPLAD-det")

ANNOTATION_FILE = DATASET_DIR / "annotations" / "instances_train.json"
IMAGE_DIR = DATASET_DIR / "train"

INSPECTION_DIR = PROJECT_DIR / "ground_truth_inspection"
OUTPUT_DIR = PROJECT_DIR / "prediction_comparison"

CONFIDENCE_THRESHOLD = 0.50


# ==================================================
# LOAD GROUND-TRUTH ANNOTATIONS
# ==================================================

print("\nLoading dataset annotations...")

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    c["id"]: c["name"]
    for c in data["categories"]
}

images = {
    img["id"]: img
    for img in data["images"]
}

annotations_by_image = defaultdict(list)

for ann in data["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)

# Original filename -> image ID and image information
filename_to_data = {}

for image_id, info in images.items():
    filename_to_data[info["file_name"]] = (image_id, info)

print("Dataset images:", len(images))
print("Ground-truth annotations:", len(data["annotations"]))
print("Classes:", len(categories))


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

print("\nLoading Faster R-CNN model...")

model, metadata = load_model()

# Model class names
model_info = metadata

print("Model loaded successfully.")
print("Checkpoint:", metadata["checkpoint_path"])
print("Device:", metadata["device"])
print("Confidence threshold:", CONFIDENCE_THRESHOLD)


# ==================================================
# IMAGE DRAWING FUNCTIONS
# ==================================================

def get_font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def draw_box(draw, box, label, color, font, width=4):

    x1, y1, x2, y2 = [
        int(round(float(v))) for v in box
    ]

    draw.rectangle(
        (x1, y1, x2, y2),
        outline=color,
        width=width
    )

    draw.text(
        (max(0, x1), max(0, y1 - 22)),
        label,
        fill=color,
        font=font
    )


# ==================================================
# PREPARE OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect inspection images.
# The filenames have prefixes such as:
# 1_32-1_DJI_0673.jpg
#
# The original dataset filename is:
# 32-1_DJI_0673.jpg

selected_files = sorted(
    (
        p.name,
        p.name.split("_", 1)[1]
    )
    for p in INSPECTION_DIR.rglob("*.jpg")
    if "_" in p.name
)

print("\nInspection images found:", len(selected_files))


# ==================================================
# PROCESS IMAGES
# ==================================================

csv_rows = []
processed = 0
skipped = 0

font = get_font()

for display_filename, original_filename in selected_files:

    print("\nProcessing:", display_filename)

    # Match using original dataset filename
    if original_filename not in filename_to_data:

        print("Skipping unknown image:", original_filename)
        skipped += 1
        continue

    image_id, image_info = filename_to_data[original_filename]

    image_path = IMAGE_DIR / original_filename

    if not image_path.exists():

        print("Missing original image:", image_path)
        skipped += 1
        continue

    # Load original image
    original = Image.open(image_path).convert("RGB")

    # ==================================================
    # RUN MODEL PREDICTION
    # ==================================================

    boxes, labels, scores, elapsed = detect(
        model,
        original,
        threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE
    )

    # ==================================================
    # DRAW GROUND TRUTH AND PREDICTIONS
    # ==================================================

    output = original.copy()
    draw = ImageDraw.Draw(output)

    # ----------------------------------------------
    # GROUND TRUTH: RED
    # ----------------------------------------------

    ground_truth_annotations = annotations_by_image[image_id]

    for ann in ground_truth_annotations:

        class_name = categories[ann["category_id"]]

        x, y, w, h = ann["bbox"]

        gt_box = (
            x,
            y,
            x + w,
            y + h
        )

        draw_box(
            draw,
            gt_box,
            f"GT: {class_name}",
            "red",
            font,
            width=4
        )

    # ----------------------------------------------
    # MODEL PREDICTIONS: BLUE
    # ----------------------------------------------

    for box, label, score in zip(boxes, labels, scores):

        class_id = int(label)

        class_name = model_info["class_names"][class_id]

        confidence = float(score) * 100

        draw_box(
            draw,
            box,
            f"PRED: {class_name} {confidence:.1f}%",
            "#0066FF",
            font,
            width=3
        )

        csv_rows.append({

            "image": original_filename,
            "inspection_image": display_filename,
            "predicted_class": class_name,
            "class_id": class_id,
            "confidence": round(confidence, 2),

            "x1": round(float(box[0]), 2),
            "y1": round(float(box[1]), 2),
            "x2": round(float(box[2]), 2),
            "y2": round(float(box[3]), 2),

        })

    # ==================================================
    # SAVE ANNOTATED IMAGE
    # ==================================================

    output_path = OUTPUT_DIR / display_filename

    output.save(output_path, quality=95)

    processed += 1

    print(
        f"[{processed}] {original_filename} | "
        f"GT: {len(ground_truth_annotations)} | "
        f"Predictions: {len(boxes)} | "
        f"Time: {elapsed:.3f}s"
    )


# ==================================================
# SAVE PREDICTION CSV
# ==================================================

csv_path = OUTPUT_DIR / "predictions.csv"

fieldnames = [
    "image",
    "inspection_image",
    "predicted_class",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(csv_rows)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n===================================")
print("COMPARISON COMPLETED")
print("===================================")

print("Images found:", len(selected_files))
print("Processed images:", processed)
print("Skipped images:", skipped)
print("Prediction records:", len(csv_rows))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nCSV report:")
print(csv_path)

print("\nBOX COLOR LEGEND")
print("RED  = Ground Truth")
print("BLUE = Model Prediction")

print("\nConfidence threshold:", CONFIDENCE_THRESHOLD)
