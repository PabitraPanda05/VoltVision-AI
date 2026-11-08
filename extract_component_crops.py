
import json
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image

DATASET = Path(r"D:\python\InsPLAD\InsPLAD-det")
ANNOTATION_FILE = DATASET / "annotations" / "instances_train.json"
IMAGE_DIR = DATASET / "train"

OUTPUT_DIR = Path(r"D:\python\PowerLine_AI\dataset_inspection")

TARGET_CLASSES = [
    "lightning rod shackle",
    "glass insulator big shackle",
    "glass insulator small shackle",
    "glass insulator tower shackle",
    "polymer insulator tower shackle",
    "sphere"
]

SAMPLES_PER_CLASS = 20
PADDING_RATIO = 0.10

random.seed(42)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    c["id"]: c["name"].lower()
    for c in data["categories"]
}

images = {
    img["id"]: img
    for img in data["images"]
}

grouped = defaultdict(list)

for ann in data["annotations"]:
    class_name = categories.get(ann["category_id"])

    if class_name in TARGET_CLASSES:
        grouped[class_name].append(ann)

for class_name in TARGET_CLASSES:

    annotations = grouped[class_name]

    selected = random.sample(
        annotations,
        min(SAMPLES_PER_CLASS, len(annotations))
    )

    class_folder = OUTPUT_DIR / class_name.replace(" ", "_")
    class_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {class_name}")
    print(f"Selected crops: {len(selected)}")

    for index, ann in enumerate(selected, start=1):

        image_info = images[ann["image_id"]]
        image_path = IMAGE_DIR / image_info["file_name"]

        if not image_path.exists():
            print("Missing:", image_path)
            continue

        image = Image.open(image_path).convert("RGB")

        x, y, w, h = ann["bbox"]

        pad_x = w * PADDING_RATIO
        pad_y = h * PADDING_RATIO

        left = max(0, int(x - pad_x))
        top = max(0, int(y - pad_y))
        right = min(image.width, int(x + w + pad_x))
        bottom = min(image.height, int(y + h + pad_y))

        crop = image.crop((left, top, right, bottom))

        output_path = class_folder / f"{index:03d}.jpg"
        crop.save(output_path, quality=95)

    print("Saved to:", class_folder)

print("\nComponent crop extraction completed.")
print("Output:", OUTPUT_DIR)
