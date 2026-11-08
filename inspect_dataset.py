
import json
import random
from pathlib import Path
from collections import defaultdict

DATASET = Path(r"D:\python\InsPLAD\InsPLAD-det")

TRAIN_JSON = DATASET / "annotations" / "instances_train.json"
VAL_JSON = DATASET / "annotations" / "instances_val.json"

TARGET_CLASSES = [
    "lightning rod shackle",
    "glass insulator big shackle",
    "glass insulator small shackle",
    "glass insulator tower shackle",
    "polymer insulator tower shackle",
    "sphere"
]

def inspect_annotations(json_path, split):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {
        c["id"]: c["name"]
        for c in data["categories"]
    }

    images = {
        img["id"]: img
        for img in data["images"]
    }

    grouped = defaultdict(list)

    for ann in data["annotations"]:

        name = categories.get(ann["category_id"], "")

        if name.lower() in TARGET_CLASSES:
            grouped[name.lower()].append(ann)

    print(f"\n{'=' * 65}")
    print(f"{split.upper()} DATASET IMAGE INSPECTION")
    print("=" * 65)

    for class_name in TARGET_CLASSES:

        annotations = grouped[class_name]

        print(f"\nCLASS: {class_name}")
        print("Annotations:", len(annotations))

        if not annotations:
            print("No annotations available.")
            continue

        image_ids = list(set(
            ann["image_id"] for ann in annotations
        ))

        print("Unique images:", len(image_ids))

        samples = random.sample(
            annotations,
            min(5, len(annotations))
        )

        print("Sample bounding boxes:")

        for ann in samples:

            x, y, w, h = ann["bbox"]

            print(
                "Image:", images[ann["image_id"]]["file_name"],
                "| Width:", round(w, 2),
                "| Height:", round(h, 2),
                "| Area:", round(w * h, 2)
            )


random.seed(42)

inspect_annotations(TRAIN_JSON, "train")
inspect_annotations(VAL_JSON, "validation")
