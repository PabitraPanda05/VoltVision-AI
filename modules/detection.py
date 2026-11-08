"""Image preprocessing, inference and annotation for PowerLine_AI."""
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from config import CLASS_NAMES, DEVICE


def preprocess_image(image):
    """Convert a PIL image to an RGB float tensor in [0, 1]."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")

    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0

    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def detect(model, image, threshold=0.50, device=DEVICE):
    """
    Run Faster R-CNN inference.

    Returns:
        boxes: ndarray [N, 4] in xyxy format
        labels: ndarray [N]
        scores: ndarray [N]
        inference_seconds: float
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    tensor = preprocess_image(image).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.inference_mode():
        output = model([tensor])[0]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    boxes = output["boxes"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()

    keep = scores >= float(threshold)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # Guard against unexpected class IDs before UI code indexes CLASS_NAMES.
    valid = (labels >= 0) & (labels < len(CLASS_NAMES))
    return boxes[valid], labels[valid], scores[valid], elapsed


def annotate(image, boxes, labels, scores):
    """Draw class names, confidence values and xyxy boxes on a copy."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        class_id = int(label)
        if not 0 <= class_id < len(CLASS_NAMES):
            continue

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        name = CLASS_NAMES[class_id]
        text = f"{name}  {float(score) * 100:.1f}%"

        draw.rectangle((x1, y1, x2, y2), outline="#1699ff", width=4)

        try:
            left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            text_width, text_height = len(text) * 8, 18

        text_y = max(0, y1 - text_height - 8)
        draw.rounded_rectangle(
            (x1, text_y, x1 + text_width + 12, text_y + text_height + 8),
            radius=4,
            fill="#087ff5",
        )
        draw.text(
            (x1 + 6, text_y + 4),
            text,
            fill="white",
            font=font,
        )

    return output
