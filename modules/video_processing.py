"""Optimized video processing for PowerLine_AI.

Uses the existing Faster R-CNN model and modules.detection functions.
This is frame-based component detection; it does not perform tracking or defect classification.
"""

import csv
import io
import time
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from config import CLASS_NAMES
from modules.detection import detect, annotate


def process_video(
    input_path,
    output_path,
    model,
    confidence_threshold=0.50,
    frame_interval=20,
    progress_callback=None,
    inference_max_width=1280,
):
    """Process a video and save an annotated MP4 plus detection CSV data.

    Runs inference on the first frame and then every ``frame_interval`` frames.
    Frames between samples are copied unchanged to the output video.

    ``inference_max_width`` downsizes only the image sent to the detector.
    Predictions are scaled back to the original video dimensions for annotation.
    Set it to None to disable inference resizing.
    """
    confidence_threshold = float(confidence_threshold)
    frame_interval = int(frame_interval)

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if frame_interval < 1:
        raise ValueError("frame_interval must be at least 1")
    if inference_max_width is not None and int(inference_max_width) < 320:
        raise ValueError("inference_max_width must be at least 320 or None")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(
            "OpenCV could not open this video. Try converting it to a standard MP4 (H.264) or AVI file."
        )

    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("The video has invalid frame dimensions.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                "OpenCV could not create the output MP4. Check codec support and output path."
            )

        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow([
            "frame_number", "timestamp_seconds", "component", "confidence",
            "x1", "y1", "x2", "y2",
        ])

        component_counts = Counter()
        total_detections = 0
        analyzed_frames = 0
        processed_frames = 0
        inference_total = 0.0
        started = time.perf_counter()

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_number = processed_frames + 1
            output_frame = frame_bgr

            if (frame_number - 1) % frame_interval == 0:
                original_h, original_w = frame_bgr.shape[:2]
                inference_frame = frame_bgr
                scale_x = scale_y = 1.0

                if inference_max_width and original_w > int(inference_max_width):
                    target_w = int(inference_max_width)
                    target_h = max(1, int(round(original_h * target_w / original_w)))
                    inference_frame = cv2.resize(
                        frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
                    )
                    scale_x = original_w / target_w
                    scale_y = original_h / target_h

                frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                boxes, labels, scores, inference_seconds = detect(
                    model, pil_image, confidence_threshold
                )
                inference_total += float(inference_seconds)
                analyzed_frames += 1

                # Map resized-image coordinates back to original video coordinates.
                if scale_x != 1.0 or scale_y != 1.0:
                    boxes = [
                        [
                            float(box[0]) * scale_x,
                            float(box[1]) * scale_y,
                            float(box[2]) * scale_x,
                            float(box[3]) * scale_y,
                        ]
                        for box in boxes
                    ]

                original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                original_image = Image.fromarray(original_rgb)
                annotated_image = annotate(original_image, boxes, labels, scores)
                output_frame = cv2.cvtColor(
                    np.asarray(annotated_image), cv2.COLOR_RGB2BGR
                )

                timestamp_seconds = (frame_number - 1) / fps
                for box, label, score in zip(boxes, labels, scores):
                    class_id = int(label)
                    component_name = (
                        CLASS_NAMES[class_id]
                        if 0 <= class_id < len(CLASS_NAMES)
                        else f"Class_{class_id}"
                    )
                    csv_writer.writerow([
                        frame_number,
                        f"{timestamp_seconds:.3f}",
                        component_name,
                        f"{float(score):.6f}",
                        int(round(float(box[0]))),
                        int(round(float(box[1]))),
                        int(round(float(box[2]))),
                        int(round(float(box[3]))),
                    ])
                    component_counts[component_name] += 1
                    total_detections += 1

            writer.write(output_frame)
            processed_frames += 1

            if progress_callback and (
                processed_frames == 1
                or processed_frames % 10 == 0
                or (reported_frames > 0 and processed_frames >= reported_frames)
            ):
                if reported_frames > 0:
                    progress = min(processed_frames / reported_frames, 1.0)
                    message = (
                        f"Processing frame {processed_frames:,} of {reported_frames:,} "
                        f"· analyzed {analyzed_frames:,}"
                    )
                else:
                    progress = 0.0
                    message = (
                        f"Processed {processed_frames:,} frames "
                        f"· analyzed {analyzed_frames:,}"
                    )
                progress_callback(progress, message)

        processing_seconds = time.perf_counter() - started
        if processed_frames == 0:
            raise RuntimeError("The uploaded video contains no readable frames.")

        return {
            "csv_text": csv_buffer.getvalue(),
            "summary": {
                "total_frames": processed_frames,
                "analyzed_frames": analyzed_frames,
                "total_detections": total_detections,
                "processing_seconds": processing_seconds,
                "average_inference_seconds": (
                    inference_total / analyzed_frames if analyzed_frames else 0.0
                ),
                "component_counts": dict(component_counts),
                "fps": fps,
                "width": width,
                "height": height,
                "frame_interval": frame_interval,
                "inference_max_width": inference_max_width,
            },
        }
    finally:
        capture.release()
        if writer is not None:
            writer.release()
