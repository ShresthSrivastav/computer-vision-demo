from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import demo_dataset_zip, model_artifact_bytes, parse_dataset_zip, predict_images, split_dataset, train_classifier


def main() -> None:
    dataset = parse_dataset_zip(demo_dataset_zip())
    splits = split_dataset(dataset["images"])
    result = train_classifier(
        dataset["images"],
        splits,
        "MobileNetV3-Small",
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
    )
    assert len(result["labels"]) == 5
    assert len(result["history"]) == 1
    assert len(result["predictions"]) == len(splits["test"])
    assert 0 <= result["metrics"]["accuracy"] <= 1
    sample = dataset["images"][0]
    assert len(predict_images(result, [(sample["name"], sample["data"])])) == 1
    assert model_artifact_bytes(result).startswith(b"PK")
    print(f"CPU training smoke check passed · test accuracy {result['metrics']['accuracy']:.1%}")


if __name__ == "__main__":
    main()
