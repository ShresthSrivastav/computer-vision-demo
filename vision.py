"""CPU-first computer-vision inference helpers for Vision Studio."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

TASK_CLASSIFICATION = "Image classification"
TASK_DETECTION = "Object detection"
TASK_SEGMENTATION = "Semantic segmentation"
TASK_OCR = "Text recognition (OCR)"
TASKS = (TASK_CLASSIFICATION, TASK_DETECTION, TASK_SEGMENTATION, TASK_OCR)


class ImageValidationError(ValueError):
    """An uploaded image is invalid or exceeds a documented limit."""


@dataclass
class ModelBundle:
    task: str
    model: Any
    weights: Any = None
    transform: Any = None
    categories: Sequence[str] = ()


def validate_image(
    data: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> Image.Image:
    """Decode a supported image safely and return an EXIF-corrected RGB copy."""
    if not data:
        raise ImageValidationError("The selected file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"Image files must be {max_bytes // (1024 * 1024)} MB or smaller."
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise ImageValidationError("Use a JPEG, PNG, or WebP image.")
            width, height = probe.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ImageValidationError(
                    "This image has too many pixels to process. Choose an image under "
                    f"{max_pixels:,} pixels."
                )
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            corrected = ImageOps.exif_transpose(opened)
            return corrected.convert("RGB").copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            "This file could not be decoded as a valid image. Try another JPEG, PNG, or WebP."
        ) from exc


def image_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_download_stem(filename: str) -> str:
    leaf = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf.rsplit(".", 1)[0])
    return stem.strip("._-")[:80] or "vision_result"


def build_model(task: str) -> ModelBundle:
    """Build only the requested CPU model; Streamlit caches one bundle at a time."""
    if task not in TASKS:
        raise ValueError(f"Unknown vision task: {task}")

    if task == TASK_OCR:
        from rapidocr import RapidOCR

        return ModelBundle(task=task, model=RapidOCR())

    import torch

    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))

    if task == TASK_CLASSIFICATION:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights).eval()
        return ModelBundle(task, model, weights, weights.transforms(), weights.meta["categories"])

    if task == TASK_DETECTION:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval()
        return ModelBundle(task, model, weights, None, weights.meta["categories"])

    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
        lraspp_mobilenet_v3_large,
    )

    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights).eval()
    return ModelBundle(task, model, weights, weights.transforms(), weights.meta["categories"])


def classify(bundle: ModelBundle, image: Image.Image, top_k: int = 5) -> list[dict[str, Any]]:
    import torch

    batch = bundle.transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(bundle.model(batch)[0], dim=0)
        count = min(max(1, int(top_k)), len(bundle.categories))
        scores, indices = torch.topk(probabilities, count)
    return [
        {"rank": rank, "label": bundle.categories[int(index)], "confidence": float(score)}
        for rank, (score, index) in enumerate(zip(scores, indices), start=1)
    ]


def detect(
    bundle: ModelBundle,
    image: Image.Image,
    threshold: float = 0.55,
    max_detections: int = 25,
) -> list[dict[str, Any]]:
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    tensor = pil_to_tensor(image.convert("RGB")).float().div_(255)
    with torch.inference_mode():
        prediction = bundle.model([tensor])[0]

    detections: list[dict[str, Any]] = []
    for box, label_index, score in zip(
        prediction["boxes"].cpu().tolist(),
        prediction["labels"].cpu().tolist(),
        prediction["scores"].cpu().tolist(),
    ):
        if score < threshold:
            continue
        label = int(label_index)
        if not 0 <= label < len(bundle.categories):
            continue
        detections.append(
            {
                "label": bundle.categories[label],
                "confidence": float(score),
                "box": [float(value) for value in box],
            }
        )
    return detections[:max_detections]


def segment(bundle: ModelBundle, image: Image.Image) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import torch

    rgb = image.convert("RGB")
    tensor = bundle.transform(rgb).unsqueeze(0)
    with torch.inference_mode():
        logits = bundle.model(tensor)["out"][0]
        small_mask = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)

    mask = np.asarray(
        Image.fromarray(small_mask).resize(rgb.size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    counts = np.bincount(mask.reshape(-1), minlength=len(bundle.categories))
    total = int(mask.size)
    regions = [
        {
            "label": bundle.categories[index],
            "pixels": int(count),
            "share": float(count / total),
            "color": color_for_label(index),
        }
        for index, count in enumerate(counts)
        if index > 0 and count > 0 and index < len(bundle.categories)
    ]
    regions.sort(key=lambda row: row["pixels"], reverse=True)
    return mask, regions


def recognize_text(bundle: ModelBundle, image: Image.Image) -> list[dict[str, Any]]:
    result = bundle.model(image.convert("RGB"))
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None or scores is None:
        return []

    lines: list[dict[str, Any]] = []
    for text, score, box in zip(texts, scores, boxes):
        clean_text = str(text).strip()
        if clean_text:
            lines.append(
                {
                    "text": clean_text,
                    "confidence": float(score),
                    "box": np.asarray(box, dtype=float).tolist(),
                }
            )
    return lines


def color_for_label(label: int) -> tuple[int, int, int]:
    """Return a deterministic, readable color without a plotting dependency."""
    hue = (label * 0.618033988749895) % 1.0
    red, green, blue = _hsv_to_rgb(hue, 0.68, 0.96)
    return round(red * 255), round(green * 255), round(blue * 255)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    import colorsys

    return colorsys.hsv_to_rgb(hue, saturation, value)


def draw_detection_overlay(
    image: Image.Image, detections: Iterable[dict[str, Any]]
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    width, height = output.size
    line_width = max(2, round(min(width, height) / 300))

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        x1, x2 = sorted((max(0, min(width - 1, round(x1))), max(0, min(width - 1, round(x2)))))
        y1, y2 = sorted((max(0, min(height - 1, round(y1))), max(0, min(height - 1, round(y2)))))
        color = color_for_label(index)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{detection['label']}  {detection['confidence']:.0%}"
        text_box = draw.textbbox((0, 0), label, font=font)
        label_height = text_box[3] - text_box[1]
        label_top = max(0, y1 - label_height - 7)
        draw.rectangle((x1, label_top, x1 + (text_box[2] - text_box[0]) + 8, label_top + label_height + 6), fill=color)
        draw.text((x1 + 4, label_top + 3), label, fill="#07111f", font=font)
    return output


def draw_segmentation_overlay(image: Image.Image, mask: np.ndarray, alpha: int = 145) -> Image.Image:
    rgb = image.convert("RGB")
    overlay_data = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    for label in np.unique(mask):
        if int(label) == 0:
            continue
        overlay_data[mask == label, :3] = color_for_label(int(label))
        overlay_data[mask == label, 3] = alpha
    overlay = Image.fromarray(overlay_data, mode="RGBA")
    return Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")


def draw_segmentation_mask(mask: np.ndarray) -> Image.Image:
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in np.unique(mask):
        if int(label) > 0:
            colors[mask == label] = color_for_label(int(label))
    return Image.fromarray(colors, mode="RGB")


def draw_ocr_overlay(image: Image.Image, lines: Iterable[dict[str, Any]]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    line_width = max(2, round(min(output.size) / 300))
    for line in lines:
        points = [(round(point[0]), round(point[1])) for point in line["box"]]
        if len(points) < 4:
            continue
        draw.polygon(points, outline="#9de6c3", width=line_width)
        x, y = points[0]
        label = f"{line['text'][:28]}  {line['confidence']:.0%}"
        text_box = draw.textbbox((0, 0), label, font=font)
        label_height = text_box[3] - text_box[1]
        label_top = max(0, y - label_height - 7)
        draw.rectangle((x, label_top, x + (text_box[2] - text_box[0]) + 8, label_top + label_height + 6), fill="#9de6c3")
        draw.text((x + 4, label_top + 3), label, fill="#07111f", font=font)
    return output


def rows_to_csv(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(_safe_csv_row(headers))
    writer.writerows(_safe_csv_row(row) for row in rows)
    return stream.getvalue().encode("utf-8-sig")


def _safe_csv_row(row: Sequence[Any]) -> list[Any]:
    safe: list[Any] = []
    for value in row:
        if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
            safe.append("'" + value)
        else:
            safe.append(value)
    return safe


def png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()
