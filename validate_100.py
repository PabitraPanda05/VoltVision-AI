
from pathlib import Path
import json
import random
import csv
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from config import CLASS_NAMES, DEVICE, MODEL_PATH
from modules.model_loader import load_model


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_DIR = Path(r"D:\python\InsPLAD\InsPLAD-det")
ANNOTATION_FILE = DATASET_DIR / "annotations" / "instances_val.json"
IMAGE_DIR = DATASET_DIR / "val"

NUM_IMAGES = 2626
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
RANDOM_SEED = 42

OUTPUT_CSV = Path("validation_100_results.csv")


# =====================================================
# IOU CALCULATION
# =====================================================

def calculate_iou(box1, box2):
    """
    Both boxes use xyxy format:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 65)
print("POWERLINE AI - 100 IMAGE VALIDATION")
print("=" * 65)

if not ANNOTATION_FILE.exists():
    raise FileNotFoundError(ANNOTATION_FILE)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

images = dataset["images"]
annotations = dataset["annotations"]

category_to_label = {
    category["id"]: category["id"]
    for category in dataset["categories"]
}

# Ground truth annotations indexed by image ID
ground_truth = defaultdict(list)

for ann in annotations:
    category_id = ann["category_id"]

    if category_id not in category_to_label:
        continue

    x, y, width, height = ann["bbox"]

    box = [
        x,
        y,
        x + width,
        y + height
    ]

    ground_truth[ann["image_id"]].append({
        "label": category_to_label[category_id],
        "box": box
    })


# =====================================================
# SELECT VALIDATION IMAGES
# =====================================================

random.seed(RANDOM_SEED)

available_images = [
    img for img in images
    if (IMAGE_DIR / img["file_name"]).exists()
]

if len(available_images) < NUM_IMAGES:
    print(
        f"Only {len(available_images)} images are available."
    )

selected_images = random.sample(
    available_images,
    min(NUM_IMAGES, len(available_images))
)

print("\nTotal validation images:", len(available_images))
print("Selected images:", len(selected_images))


# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading trained model...")

model, metadata = load_model()
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)
print("Checkpoint:", MODEL_PATH)


# =====================================================
# EVALUATION COUNTERS
# =====================================================

class_stats = defaultdict(
    lambda: {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "IoU_sum": 0.0,
        "matched": 0
    }
)

total_predictions = 0
processed_images = 0


# =====================================================
# RUN INFERENCE
# =====================================================

print("\nStarting validation...\n")

original_score_threshold = model.roi_heads.score_thresh
original_max_detections = model.roi_heads.detections_per_img

# Allow lower-scoring candidates through model inference.
# We apply the actual evaluation threshold ourselves.
model.roi_heads.score_thresh = 0.001
model.roi_heads.detections_per_img = max(
    original_max_detections, 300
)

try:
    for index, image_info in enumerate(selected_images, start=1):

        image_path = IMAGE_DIR / image_info["file_name"]
        image_id = image_info["id"]

        try:
            image = Image.open(image_path).convert("RGB")

            image_tensor = (
                torch.from_numpy(np.array(image))
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .to(DEVICE)
            )

            with torch.no_grad():
                prediction = model([image_tensor])[0]

            pred_boxes = prediction["boxes"].detach().cpu().numpy()
            pred_labels = prediction["labels"].detach().cpu().numpy()
            pred_scores = prediction["scores"].detach().cpu().numpy()

            predictions = []

            for box, label, score in zip(
                pred_boxes, pred_labels, pred_scores
            ):
                if score < CONFIDENCE_THRESHOLD:
                    continue

                predictions.append({
                    "label": int(label),
                    "box": box.tolist(),
                    "score": float(score)
                })

            actual_objects = ground_truth[image_id]

            total_predictions += len(predictions)

            # Process each foreground class
            for class_id in range(1, len(CLASS_NAMES)):

                gt_objects = [
                    obj for obj in actual_objects
                    if obj["label"] == class_id
                ]

                pred_objects = [
                    obj for obj in predictions
                    if obj["label"] == class_id
                ]

                # Highest-confidence predictions first
                pred_objects.sort(
                    key=lambda x: x["score"],
                    reverse=True
                )

                matched_gt = set()

                for pred in pred_objects:

                    best_iou = 0.0
                    best_gt_index = None

                    for gt_index, gt in enumerate(gt_objects):

                        if gt_index in matched_gt:
                            continue

                        iou = calculate_iou(
                            pred["box"],
                            gt["box"]
                        )

                        if iou > best_iou:
                            best_iou = iou
                            best_gt_index = gt_index

                    if (
                        best_gt_index is not None
                        and best_iou >= IOU_THRESHOLD
                    ):
                        class_stats[class_id]["TP"] += 1
                        class_stats[class_id]["IoU_sum"] += best_iou
                        class_stats[class_id]["matched"] += 1

                        matched_gt.add(best_gt_index)

                    else:
                        class_stats[class_id]["FP"] += 1

                class_stats[class_id]["FN"] += (
                    len(gt_objects) - len(matched_gt)
                )

            processed_images += 1

            if index % 10 == 0 or index == len(selected_images):
                print(
                    f"Processed {index}/{len(selected_images)} images"
                )

        except Exception as error:
            print(
                f"Error processing {image_path.name}: {error}"
            )

finally:
    model.roi_heads.score_thresh = original_score_threshold
    model.roi_heads.detections_per_img = original_max_detections


# =====================================================
# CALCULATE METRICS
# =====================================================

print("\n" + "=" * 90)
print("CLASS-WISE VALIDATION RESULTS")
print("=" * 90)

print(
    f"{'Class':38s} {'TP':>5s} {'FP':>5s} "
    f"{'FN':>5s} {'Precision':>11s} "
    f"{'Recall':>10s} {'F1':>10s} {'Mean IoU':>10s}"
)

rows = []

total_tp = 0
total_fp = 0
total_fn = 0
total_iou = 0.0
total_matched = 0

for class_id in range(1, len(CLASS_NAMES)):

    stats = class_stats[class_id]

    tp = stats["TP"]
    fp = stats["FP"]
    fn = stats["FN"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    mean_iou = (
        stats["IoU_sum"] / stats["matched"]
        if stats["matched"] else 0.0
    )

    class_name = CLASS_NAMES[class_id]

    rows.append({
        "Class": class_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Mean_IoU": mean_iou
    })

    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_iou += stats["IoU_sum"]
    total_matched += stats["matched"]

    print(
        f"{class_name:38s} {tp:5d} {fp:5d} {fn:5d} "
        f"{precision * 100:10.2f}% "
        f"{recall * 100:9.2f}% "
        f"{f1 * 100:9.2f}% "
        f"{mean_iou * 100:9.2f}%"
    )


# =====================================================
# OVERALL MICRO METRICS
# =====================================================

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp else 0.0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

mean_iou = (
    total_iou / total_matched
    if total_matched else 0.0
)

print("\n" + "=" * 65)
print("OVERALL MICRO-AVERAGED RESULTS")
print("=" * 65)

print("Successfully processed images:", processed_images)
print("Total predictions:", total_predictions)
print("True Positives:", total_tp)
print("False Positives:", total_fp)
print("False Negatives:", total_fn)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Mean IoU: {mean_iou * 100:.2f}%")


# =====================================================
# SAVE CSV REPORT
# =====================================================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Class", "TP", "FP", "FN",
            "Precision", "Recall", "F1", "Mean_IoU"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nCSV report saved:", OUTPUT_CSV.resolve())

print("\nVALIDATION COMPLETED.")
