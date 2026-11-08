
import json
import csv
from pathlib import Path
from collections import defaultdict

from PIL import Image

from config import DEVICE
from modules.model_loader import load_model
from modules.detection import detect


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_DIR = Path(r"D:\python\PowerLine_AI")
DATASET_DIR = Path(r"D:\python\InsPLAD\InsPLAD-det")

ANNOTATION_FILE = DATASET_DIR / "annotations" / "instances_train.json"
IMAGE_DIR = DATASET_DIR / "train"
INSPECTION_DIR = PROJECT_DIR / "ground_truth_inspection"

OUTPUT_DIR = PROJECT_DIR / "detailed_evaluation"

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


# =====================================================
# IOU CALCULATION
# =====================================================

def calculate_iou(box1, box2):

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))

    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, float(box1[2]) - float(box1[0])) * max(
        0, float(box1[3]) - float(box1[1])
    )

    area2 = max(0, float(box2[2]) - float(box2[0])) * max(
        0, float(box2[3]) - float(box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# =====================================================
# LOAD ANNOTATIONS
# =====================================================

print("\nLoading dataset annotations...")

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    int(c["id"]): c["name"]
    for c in data["categories"]
}

images = {
    int(img["id"]): img
    for img in data["images"]
}

filename_to_data = {
    info["file_name"]: (image_id, info)
    for image_id, info in images.items()
}

annotations_by_image = defaultdict(list)

for ann in data["annotations"]:
    annotations_by_image[int(ann["image_id"])].append(ann)

print("Images:", len(images))
print("Annotations:", len(data["annotations"]))
print("Classes:", len(categories))


# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading trained Faster R-CNN model...")

model, metadata = load_model()

class_names = metadata["class_names"]

print("Model loaded successfully.")
print("Checkpoint:", metadata["checkpoint_path"])
print("Device:", metadata["device"])
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("IoU threshold:", IOU_THRESHOLD)


# =====================================================
# CLASS NAME HELPER
# =====================================================

def get_class_name(class_id):

    class_id = int(class_id)

    if class_id in class_names:
        return class_names[class_id]

    if 0 <= class_id < len(class_names):
        return class_names[class_id]

    return f"Unknown_{class_id}"


# =====================================================
# PREPARE OUTPUT
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

selected_files = sorted(
    (p.name, p.name.split("_", 1)[1])
    for p in INSPECTION_DIR.rglob("*.jpg")
    if "_" in p.name
)

print("\nSelected images:", len(selected_files))


# =====================================================
# METRIC STORAGE
# =====================================================

class_metrics = {
    class_id: {
        "TP": 0,
        "FP": 0,
        "FN": 0
    }
    for class_id in categories
}

detailed_rows = []

processed_images = 0
total_ground_truth = 0
total_predictions = 0
correct_matches = 0
misclassifications = 0
unmatched_predictions = 0
unmatched_ground_truth = 0


# =====================================================
# PROCESS EACH IMAGE
# =====================================================

for display_filename, original_filename in selected_files:

    print("\nProcessing:", original_filename)

    if original_filename not in filename_to_data:
        print("Unknown dataset image. Skipping.")
        continue

    image_id, image_info = filename_to_data[original_filename]

    image_path = IMAGE_DIR / original_filename

    if not image_path.exists():
        print("Image not found:", image_path)
        continue

    original = Image.open(image_path).convert("RGB")

    # ---------------------------------------------
    # GROUND TRUTH
    # ---------------------------------------------

    gt_objects = []

    for ann in annotations_by_image[image_id]:

        x, y, w, h = ann["bbox"]

        gt_objects.append({
            "class_id": int(ann["category_id"]),
            "class_name": categories[int(ann["category_id"])],
            "box": [
                x,
                y,
                x + w,
                y + h
            ]
        })

    # ---------------------------------------------
    # MODEL PREDICTIONS
    # ---------------------------------------------

    boxes, labels, scores, elapsed = detect(
        model,
        original,
        threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE
    )

    predictions = []

    for box, label, score in zip(boxes, labels, scores):

        class_id = int(label)

        predictions.append({
            "class_id": class_id,
            "class_name": get_class_name(class_id),
            "box": [float(v) for v in box],
            "confidence": float(score)
        })

    total_ground_truth += len(gt_objects)
    total_predictions += len(predictions)

    # ---------------------------------------------
    # CLASS-AGNOSTIC GREEDY MATCHING
    # Match by IoU first, then check class.
    # ---------------------------------------------

    unmatched_gt = set(range(len(gt_objects)))
    image_rows = []

    predictions_sorted = sorted(
        predictions,
        key=lambda p: p["confidence"],
        reverse=True
    )

    for pred in predictions_sorted:

        best_gt_index = None
        best_iou = 0.0

        for gt_index in unmatched_gt:

            iou = calculate_iou(
                pred["box"],
                gt_objects[gt_index]["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        # -----------------------------------------
        # NO MATCH AT REQUIRED IOU
        # -----------------------------------------

        if best_gt_index is None or best_iou < IOU_THRESHOLD:

            class_metrics[pred["class_id"]]["FP"] += 1

            unmatched_predictions += 1

            image_rows.append({
                "image": original_filename,
                "status": "FALSE_POSITIVE",
                "ground_truth_class": "",
                "predicted_class": pred["class_name"],
                "ground_truth_class_id": "",
                "predicted_class_id": pred["class_id"],
                "confidence_percent": round(
                    pred["confidence"] * 100, 2
                ),
                "IoU": round(best_iou, 4),
                "ground_truth_box": "",
                "predicted_box": str(pred["box"])
            })

            continue

        # -----------------------------------------
        # MATCH FOUND
        # -----------------------------------------

        gt = gt_objects[best_gt_index]
        unmatched_gt.remove(best_gt_index)

        if pred["class_id"] == gt["class_id"]:

            status = "TRUE_POSITIVE"

            class_metrics[gt["class_id"]]["TP"] += 1

            correct_matches += 1

        else:

            status = "MISCLASSIFICATION"

            # Wrong predicted class = FP
            class_metrics[pred["class_id"]]["FP"] += 1

            # Actual ground-truth class = FN
            class_metrics[gt["class_id"]]["FN"] += 1

            misclassifications += 1

        image_rows.append({
            "image": original_filename,
            "status": status,
            "ground_truth_class": gt["class_name"],
            "predicted_class": pred["class_name"],
            "ground_truth_class_id": gt["class_id"],
            "predicted_class_id": pred["class_id"],
            "confidence_percent": round(
                pred["confidence"] * 100, 2
            ),
            "IoU": round(best_iou, 4),
            "ground_truth_box": str(gt["box"]),
            "predicted_box": str(pred["box"])
        })

    # ---------------------------------------------
    # REMAINING GROUND-TRUTH OBJECTS = FN
    # ---------------------------------------------

    for gt_index in unmatched_gt:

        gt = gt_objects[gt_index]

        class_metrics[gt["class_id"]]["FN"] += 1

        unmatched_ground_truth += 1

        image_rows.append({
            "image": original_filename,
            "status": "FALSE_NEGATIVE",
            "ground_truth_class": gt["class_name"],
            "predicted_class": "",
            "ground_truth_class_id": gt["class_id"],
            "predicted_class_id": "",
            "confidence_percent": "",
            "IoU": "",
            "ground_truth_box": str(gt["box"]),
            "predicted_box": ""
        })

    detailed_rows.extend(image_rows)

    processed_images += 1

    print(
        f"[{processed_images}] "
        f"GT: {len(gt_objects)} | "
        f"Predictions: {len(predictions)} | "
        f"Time: {elapsed:.3f}s"
    )


# =====================================================
# CLASS-WISE METRICS
# =====================================================

summary_rows = []

print("\n")
print("=" * 100)
print("CLASS-WISE EVALUATION")
print("=" * 100)

print(
    f"{'Class':38s} "
    f"{'TP':>6s} {'FP':>6s} {'FN':>6s} "
    f"{'Precision':>11s} {'Recall':>10s} {'F1':>10s}"
)

for class_id, metrics in class_metrics.items():

    tp = metrics["TP"]
    fp = metrics["FP"]
    fn = metrics["FN"]

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    class_name = categories[class_id]

    summary_rows.append({
        "class_id": class_id,
        "class_name": class_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4)
    })

    print(
        f"{class_name:38.38s} "
        f"{tp:6d} {fp:6d} {fn:6d} "
        f"{precision * 100:10.2f}% "
        f"{recall * 100:9.2f}% "
        f"{f1 * 100:9.2f}%"
    )


# =====================================================
# OVERALL METRICS
# =====================================================

total_tp = sum(m["TP"] for m in class_metrics.values())
total_fp = sum(m["FP"] for m in class_metrics.values())
total_fn = sum(m["FN"] for m in class_metrics.values())

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0 else 0.0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0 else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0.0
)


# =====================================================
# SAVE DETAILED CSV
# =====================================================

detail_path = OUTPUT_DIR / "detailed_matches.csv"

detail_fields = [
    "image",
    "status",
    "ground_truth_class",
    "predicted_class",
    "ground_truth_class_id",
    "predicted_class_id",
    "confidence_percent",
    "IoU",
    "ground_truth_box",
    "predicted_box"
]

with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(f, fieldnames=detail_fields)
    writer.writeheader()
    writer.writerows(detailed_rows)


# =====================================================
# SAVE CLASS SUMMARY CSV
# =====================================================

summary_path = OUTPUT_DIR / "class_wise_metrics.csv"

summary_fields = [
    "class_id",
    "class_name",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1"
]

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)


# =====================================================
# FINAL REPORT
# =====================================================

print("\n")
print("=" * 60)
print("DETAILED EVALUATION COMPLETED")
print("=" * 60)

print("Images processed:", processed_images)
print("Ground-truth objects:", total_ground_truth)
print("Predictions:", total_predictions)

print("\nCorrect class matches:", correct_matches)
print("Misclassifications:", misclassifications)
print("Unmatched predictions:", unmatched_predictions)
print("Unmatched ground-truth objects:", unmatched_ground_truth)

print("\nOVERALL CLASS-AWARE METRICS")
print(f"TP: {total_tp}")
print(f"FP: {total_fp}")
print(f"FN: {total_fn}")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

print("\nOUTPUT FILES")
print("Detailed matches:", detail_path)
print("Class-wise metrics:", summary_path)

print("\nMatching configuration:")
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("IoU threshold:", IOU_THRESHOLD)

print("\nNote:")
print("A class mismatch is counted as FP for the predicted class")
print("and FN for the actual ground-truth class.")
print("These are custom metrics on the selected inspection images,")
print("not COCO mAP.")
