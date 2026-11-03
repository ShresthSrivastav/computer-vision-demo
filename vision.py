"""Dataset validation, transfer learning, evaluation, and export helpers."""

from __future__ import annotations

import csv
import io
import json
import random
import re
import zipfile
import zlib
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MAX_DATASET_BYTES = 25 * 1024 * 1024
MAX_UNPACKED_BYTES = 80 * 1024 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_ZIP_ENTRIES = 1_000
MAX_IMAGES = 250
MAX_CLASSES = 15
MIN_IMAGES_PER_CLASS = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
ARCHITECTURES = ("MobileNetV3-Small", "ResNet18")


class DatasetValidationError(ValueError):
    """The uploaded archive is not a supported labeled image dataset."""


def validate_image(
    data: bytes,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> Image.Image:
    if not data:
        raise DatasetValidationError("This image is empty.")
    if len(data) > max_bytes:
        raise DatasetValidationError("This image exceeds the 5 MB per-image limit.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in {"JPEG", "PNG", "WEBP"}:
                raise DatasetValidationError("Only JPEG, PNG, and WebP images are supported.")
            width, height = probe.size
            if not width or not height or width * height > max_pixels:
                raise DatasetValidationError(f"This image exceeds the {max_pixels:,} pixel limit.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB").copy()
    except DatasetValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise DatasetValidationError("This file is not a valid JPEG, PNG, or WebP image.") from exc


def parse_dataset_zip(data: bytes) -> dict[str, Any]:
    """Read and validate a class-folder ZIP without extracting it to disk."""
    if not data:
        raise DatasetValidationError("Choose a ZIP file to continue.")
    if len(data) > MAX_DATASET_BYTES:
        raise DatasetValidationError("The ZIP exceeds the 25 MB upload limit.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetValidationError("This file is not a readable ZIP archive.") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise DatasetValidationError(f"The ZIP has too many entries (maximum {MAX_ZIP_ENTRIES:,}).")
        total_size = sum(info.file_size for info in infos)
        if total_size > MAX_UNPACKED_BYTES:
            raise DatasetValidationError("The uncompressed ZIP contents exceed the 80 MB limit.")

        candidates: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
        seen: set[str] = set()
        for info in infos:
            name = info.filename
            if info.is_dir():
                continue
            if "\\" in name or name.startswith("/") or ".." in name.split("/"):
                raise DatasetValidationError("The ZIP contains an unsafe file path.")
            parts = PurePosixPath(name).parts
            if not parts or parts[0].endswith(":"):
                raise DatasetValidationError("The ZIP contains an unsafe file path.")
            if any(part.startswith(".") or part == "__MACOSX" for part in parts):
                continue
            if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if info.file_size > MAX_IMAGE_BYTES:
                raise DatasetValidationError(f"{PurePosixPath(name).name} exceeds the 5 MB per-image limit.")
            if name in seen:
                continue
            seen.add(name)
            candidates.append((info, parts))

        if not candidates:
            raise DatasetValidationError("No JPEG, PNG, or WebP images were found in the ZIP.")
        if len(candidates) > MAX_IMAGES:
            raise DatasetValidationError(f"The ZIP has too many images (maximum {MAX_IMAGES}).")
        if any(parts[0].casefold() in SPLIT_NAMES for _, parts in candidates):
            raise DatasetValidationError(
                "Upload one labeled dataset, not pre-split train/validation/test folders. "
                "Vision Studio creates a stratified split automatically."
            )

        # Accept both class/image.jpg and dataset/class/image.jpg layouts.
        if all(len(parts) >= 3 for _, parts in candidates) and len({parts[0] for _, parts in candidates}) == 1:
            candidates = [(info, parts[1:]) for info, parts in candidates]

        images: list[dict[str, Any]] = []
        warnings: list[str] = []
        for info, parts in candidates:
            if len(parts) < 2:
                raise DatasetValidationError("Put images inside a folder named after their class.")
            if any(part.casefold() in SPLIT_NAMES for part in parts[:-1]):
                raise DatasetValidationError(
                    "Upload one labeled dataset, not pre-split train/validation/test folders. "
                    "Vision Studio creates a stratified split automatically."
                )
            label = parts[-2].strip()
            if not label or label in {".", ".."}:
                raise DatasetValidationError("Each image needs a non-empty class folder name.")
            try:
                content = archive.read(info)
                image = validate_image(content)
            except (DatasetValidationError, OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile, zlib.error) as exc:
                if len(warnings) < 3:
                    warnings.append(f"Skipped {PurePosixPath(info.filename).name}: {exc}")
                continue
            images.append({"name": PurePosixPath(info.filename).name, "label": label, "data": content, "size": image.size})

    counts = Counter(item["label"] for item in images)
    if len(counts) < 2:
        raise DatasetValidationError("A classification dataset needs at least two class folders with valid images.")
    if len(counts) > MAX_CLASSES:
        raise DatasetValidationError(f"The dataset has too many classes (maximum {MAX_CLASSES}).")
    too_small = {label: count for label, count in counts.items() if count < MIN_IMAGES_PER_CLASS}
    if too_small:
        details = ", ".join(f"{label}: {count}" for label, count in sorted(too_small.items()))
        raise DatasetValidationError(
            f"Each class needs at least {MIN_IMAGES_PER_CLASS} valid images. Current counts: {details}."
        )
    return {
        "images": images,
        "classes": sorted(counts),
        "class_counts": dict(sorted(counts.items())),
        "warnings": warnings,
    }


def split_dataset(images: Sequence[dict[str, Any]], seed: int = 42) -> dict[str, list[int]]:
    """Create deterministic, class-stratified 70/15/15 train/validation/test indices."""
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(images):
        groups.setdefault(item["label"], []).append(index)
    if len(groups) < 2 or any(len(indices) < MIN_IMAGES_PER_CLASS for indices in groups.values()):
        raise DatasetValidationError("Each of at least two classes needs five or more images before splitting.")

    rng = random.Random(seed)
    result: dict[str, list[int]] = {name: [] for name in ("train", "validation", "test")}
    for indices in groups.values():
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * 0.15))
        test_count = max(1, round(len(indices) * 0.15))
        result["validation"].extend(indices[:validation_count])
        result["test"].extend(indices[validation_count : validation_count + test_count])
        result["train"].extend(indices[validation_count + test_count :])
    for indices in result.values():
        rng.shuffle(indices)
    return result


def classification_metrics(
    actual: Sequence[int], predicted: Sequence[int], labels: Sequence[str]
) -> dict[str, Any]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must be non-empty and have equal length.")
    count = len(labels)
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, guess in zip(actual, predicted):
        if not 0 <= int(truth) < count or not 0 <= int(guess) < count:
            raise ValueError("A prediction refers to an unknown class index.")
        matrix[int(truth)][int(guess)] += 1
    per_class = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": label, "precision": precision, "recall": recall, "f1": f1, "support": support})
    accuracy = sum(matrix[i][i] for i in range(count)) / len(actual)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": sum(row["recall"] for row in per_class) / count,
        "macro_precision": sum(row["precision"] for row in per_class) / count,
        "macro_recall": sum(row["recall"] for row in per_class) / count,
        "macro_f1": sum(row["f1"] for row in per_class) / count,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _image_dataset(images: Sequence[dict[str, Any]], indices: Sequence[int], class_to_index: dict[str, int], transform: Any) -> Any:
    import torch

    class ImageDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(indices)

        def __getitem__(self, item: int) -> tuple[Any, int]:
            image_item = images[indices[item]]
            with Image.open(io.BytesIO(image_item["data"])) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                return transform(image), class_to_index[image_item["label"]]

    return ImageDataset()


def _build_model(architecture: str, num_classes: int, *, pretrained: bool = True) -> tuple[Any, Any, tuple[float, ...], tuple[float, ...]]:
    import torch.nn as nn
    from torchvision.models import (
        MobileNet_V3_Small_Weights,
        ResNet18_Weights,
        mobilenet_v3_small,
        resnet18,
    )

    if architecture == "MobileNetV3-Small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        if weights is None:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            eval_transform = _inference_transform(mean, std)
        else:
            mean, std = weights.transforms().mean, weights.transforms().std
            eval_transform = weights.transforms()
    elif architecture == "ResNet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if weights is None:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            eval_transform = _inference_transform(mean, std)
        else:
            mean, std = weights.transforms().mean, weights.transforms().std
            eval_transform = weights.transforms()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model.cpu(), eval_transform, tuple(mean), tuple(std)


def _inference_transform(mean: Sequence[float], std: Sequence[float]) -> Any:
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def train_classifier(
    images: Sequence[dict[str, Any]],
    splits: dict[str, list[int]],
    architecture: str,
    *,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    seed: int = 42,
    on_epoch: Callable[[int, int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Train a frozen ImageNet backbone's new head, then evaluate the holdout test split."""
    import copy
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms

    if architecture not in ARCHITECTURES:
        raise ValueError(f"Choose one of: {', '.join(ARCHITECTURES)}")
    if not 1 <= epochs <= 6 or batch_size not in {4, 8, 16} or not 0 < learning_rate <= 0.01:
        raise ValueError("Training settings are outside supported limits.")
    labels = sorted({item["label"] for item in images})
    if len(labels) < 2 or any(not splits.get(name) for name in ("train", "validation", "test")):
        raise DatasetValidationError("The dataset split is incomplete. Upload at least five images per class.")
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    torch.manual_seed(seed)
    class_to_index = {label: index for index, label in enumerate(labels)}
    model, eval_transform, mean, std = _build_model(architecture, len(labels))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_data = _image_dataset(images, splits["train"], class_to_index, train_transform)
    validation_data = _image_dataset(images, splits["validation"], class_to_index, eval_transform)
    test_data = _image_dataset(images, splits["test"], class_to_index, eval_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    # Frozen features stay in eval mode, so batch-normalization statistics aren't updated on tiny datasets.
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None

    def evaluate(loader: Any, collect: bool = False) -> tuple[float, float, list[int], list[int], list[float]]:
        model.eval()
        total_loss = total = correct = 0
        actual: list[int] = []
        predicted: list[int] = []
        confidence: list[float] = []
        with torch.inference_mode():
            for batch, targets in loader:
                logits = model(batch)
                loss = loss_fn(logits, targets)
                probabilities = torch.softmax(logits, dim=1)
                scores, guesses = probabilities.max(dim=1)
                total_loss += float(loss) * len(targets)
                total += len(targets)
                correct += int((guesses == targets).sum())
                if collect:
                    actual.extend(targets.tolist())
                    predicted.extend(guesses.tolist())
                    confidence.extend(scores.tolist())
        return total_loss / total, correct / total, actual, predicted, confidence

    for epoch in range(1, epochs + 1):
        model.train()
        if architecture == "MobileNetV3-Small":
            model.features.eval()
        else:
            model.eval()
            model.fc.train()
        train_total = train_loss = train_correct = 0
        for batch, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(targets)
            train_total += len(targets)
            train_correct += int((logits.argmax(dim=1) == targets).sum())
        val_loss, val_accuracy, _, _, _ = evaluate(validation_loader)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        }
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if on_epoch:
            on_epoch(epoch, epochs, row)

    assert best_state is not None
    model.load_state_dict(best_state)
    test_loss, test_accuracy, actual, predicted, confidence = evaluate(test_loader, collect=True)
    metrics = classification_metrics(actual, predicted, labels)
    predictions = []
    for index, truth, guess, score in zip(splits["test"], actual, predicted, confidence):
        source = images[index]
        predictions.append({
            "filename": source["name"],
            "actual": labels[truth],
            "predicted": labels[guess],
            "confidence": score,
            "correct": truth == guess,
        })
    return {
        "architecture": architecture,
        "labels": labels,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "metrics": metrics,
        "predictions": predictions,
        "model_state": best_state,
        "preprocessing": {
            "color_mode": "RGB",
            "resize_short_edge": 256,
            "crop": "center",
            "crop_size": [224, 224],
            "mean": list(mean),
            "std": list(std),
        },
    }


def predict_images(run: dict[str, Any], uploads: Sequence[tuple[str, bytes]]) -> list[dict[str, Any]]:
    """Run a trained session model on new images without downloading pretrained weights again."""
    import torch

    model, transform, _, _ = _build_model(run["architecture"], len(run["labels"]), pretrained=False)
    model.load_state_dict(run["model_state"])
    model.eval()
    results = []
    with torch.inference_mode():
        for filename, content in uploads:
            image = validate_image(content)
            logits = model(transform(image).unsqueeze(0))
            probabilities = torch.softmax(logits[0], dim=0)
            score, index = probabilities.max(dim=0)
            results.append({"filename": filename, "predicted": run["labels"][int(index)], "confidence": float(score)})
    return results


def model_artifact_bytes(run: dict[str, Any]) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save({
        "format": "vision-studio-state-dict-v1",
        "architecture": run["architecture"],
        "labels": run["labels"],
        "preprocessing": run["preprocessing"],
        "state_dict": run["model_state"],
    }, buffer)
    return buffer.getvalue()


def rows_to_csv(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(_safe_csv_row(headers))
    writer.writerows(_safe_csv_row(row) for row in rows)
    return stream.getvalue().encode("utf-8-sig")


def _safe_csv_row(row: Sequence[Any]) -> list[Any]:
    return [
        "'" + value if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")) else value
        for value in row
    ]


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")


def safe_download_stem(filename: str) -> str:
    leaf = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf.rsplit(".", 1)[0])
    return stem.strip("._-")[:80] or "vision_dataset"


def demo_dataset_zip() -> bytes:
    """Build a tiny two-class dataset so visitors can try the whole workflow immediately."""
    import math

    rng = random.Random(9)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label in ("circle", "triangle"):
            for index in range(10):
                image = Image.new("RGB", (160, 160), (238 + rng.randrange(0, 15), 241, 245))
                draw = ImageDraw.Draw(image)
                center_x, center_y = 80 + rng.randrange(-18, 19), 80 + rng.randrange(-18, 19)
                radius = rng.randrange(28, 50)
                color = (31, 130 + rng.randrange(0, 70), 178 + rng.randrange(0, 60)) if label == "circle" else (205 + rng.randrange(0, 40), 86, 68)
                if label == "circle":
                    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=color)
                else:
                    points = [
                        (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))
                        for angle in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
                    ]
                    draw.polygon(points, fill=color)
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG", quality=88)
                archive.writestr(f"shapes/{label}/{label}_{index + 1:02d}.jpg", image_bytes.getvalue())
    return output.getvalue()
