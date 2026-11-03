"""Run each real pretrained model once on the bundled sample images."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import (
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    TASK_OCR,
    TASK_SEGMENTATION,
    build_model,
    classify,
    detect,
    recognize_text,
    segment,
    validate_image,
)


def main() -> None:
    dog = validate_image((ROOT / "assets" / "golden_retriever.jpg").read_bytes())
    text_card = validate_image((ROOT / "assets" / "ocr_sample.png").read_bytes())

    bundle = build_model(TASK_CLASSIFICATION)
    predictions = classify(bundle, dog)
    assert predictions and all(0 <= row["confidence"] <= 1 for row in predictions)
    print(f"Classification: {predictions[0]['label']} ({predictions[0]['confidence']:.1%})")
    del bundle
    gc.collect()

    bundle = build_model(TASK_DETECTION)
    detections = detect(bundle, dog, threshold=0.25)
    assert all(row["box"][2] >= row["box"][0] for row in detections)
    print(f"Detection: {len(detections)} box(es)")
    del bundle
    gc.collect()

    bundle = build_model(TASK_SEGMENTATION)
    mask, regions = segment(bundle, dog)
    assert mask.shape == (dog.height, dog.width)
    print(f"Segmentation: {len(regions)} foreground region(s)")
    del bundle
    gc.collect()

    bundle = build_model(TASK_OCR)
    lines = recognize_text(bundle, text_card)
    recognized = " ".join(row["text"] for row in lines)
    assert recognized.strip(), "OCR should recognize the bundled high-contrast text card"
    print(f"OCR: {recognized}")


if __name__ == "__main__":
    main()
