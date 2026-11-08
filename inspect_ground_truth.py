
import json
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

DATASET = Path(r"D:\python\InsPLAD\InsPLAD-det")
ANNOTATION_FILE = DATASET / "annotations" / "instances_train.json"
IMAGE_DIR = DATASET / "train"

OUTPUT_DIR = Path(r"D:\python\PowerLine_AI\ground_truth_inspection")

TARGET_CLASSES = [
    "lightning rod shackle",
    "glass insulator big shackle",
    "glass insulator small shackle",
    "glass insulator tower shackle",
]

SAMPLES_PER_CLASS = 3
random.seed(42)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {c["id"]: c["name"] for c in data["categories"]}
images = {img["id"]: img for img in data["images"]}

annotations_by_image = defaultdict(list)
images_by_class = defaultdict(set)

for ann in data["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)
    name = categories[ann["category_id"]].lower()
    if name in TARGET_CLASSES:
        images_by_class[name].add(ann["image_id"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for target in TARGET_CLASSES:
    selected_ids = random.sample(
        sorted(images_by_class[target]),
        min(SAMPLES_PER_CLASS, len(images_by_class[target]))
    )

    class_dir = OUTPUT_DIR / target.replace(" ", "_")
    class_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{target}: {len(selected_ids)} images")

    for index, image_id in enumerate(selected_ids, 1):
        info = images[image_id]
        image_path = IMAGE_DIR / info["file_name"]

        if not image_path.exists():
            print("Missing image:", image_path)
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except OSError:
            font = ImageFont.load_default()

        for ann in annotations_by_image[image_id]:
            name = categories[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            box = (x, y, x + w, y + h)
            is_target = name.lower() == target

            color = "red" if is_target else "yellow"
            width = 5 if is_target else 2

            draw.rectangle(box, outline=color, width=width)
            draw.text(
                (max(0, x), max(0, y - 25)),
                name,
                fill=color,
                font=font
            )

        output_path = class_dir / f"{index}_{info['file_name']}"
        img.save(output_path)

        print("Saved:", output_path.name)

print("\nCompleted!")
print("Output folder:", OUTPUT_DIR)
print("RED = target class | YELLOW = other annotated classes")
