from __future__ import annotations

import csv
import io
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image

from vision import (
    DatasetValidationError,
    MAX_IMAGE_BYTES,
    classification_metrics,
    demo_dataset_zip,
    parse_dataset_zip,
    rows_to_csv,
    safe_download_stem,
    split_dataset,
    validate_image,
)


def image_bytes(color: tuple[int, int, int] = (45, 100, 165), size: tuple[int, int] = (24, 20)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def dataset_zip(*, root: str = "", invalid: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for label in ("cats", "dogs"):
            for index in range(6):
                path = f"{root}{label}/{label}_{index}.jpg"
                archive.writestr(path, b"bad image" if invalid and index == 0 else image_bytes())
    return output.getvalue()


class DatasetArchiveTests(unittest.TestCase):
    def test_parses_class_folders_and_optional_wrapper(self) -> None:
        for root in ("", "collection/"):
            with self.subTest(root=root):
                dataset = parse_dataset_zip(dataset_zip(root=root))
                self.assertEqual(dataset["classes"], ["cats", "dogs"])
                self.assertEqual(dataset["class_counts"], {"cats": 6, "dogs": 6})
                self.assertEqual(len(dataset["images"]), 12)

    def test_skips_corrupt_images_with_clear_warning(self) -> None:
        parsed = parse_dataset_zip(dataset_zip(invalid=True))
        self.assertEqual(len(parsed["images"]), 10)
        self.assertEqual(len(parsed["warnings"]), 2)
        self.assertIn("Skipped", parsed["warnings"][0])

    def test_rejects_empty_invalid_and_oversize_archives(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "Choose a ZIP"):
            parse_dataset_zip(b"")
        with self.assertRaisesRegex(DatasetValidationError, "readable ZIP"):
            parse_dataset_zip(b"not a zip")
        with self.assertRaisesRegex(DatasetValidationError, "50 MB"):
            parse_dataset_zip(b"x" * (50 * 1024 * 1024 + 1))

    def test_rejects_too_much_uncompressed_data(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("notes.txt", "x" * 100)
        with patch("vision.MAX_UNPACKED_BYTES", 50), self.assertRaisesRegex(DatasetValidationError, "uncompressed"):
            parse_dataset_zip(output.getvalue())

    def test_rejects_path_traversal_and_pre_split_archives(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("../cats/escape.jpg", image_bytes())
        with self.assertRaisesRegex(DatasetValidationError, "unsafe file path"):
            parse_dataset_zip(output.getvalue())

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for label in ("cats", "dogs"):
                for index in range(5):
                    archive.writestr(f"train/{label}/{index}.jpg", image_bytes())
        with self.assertRaisesRegex(DatasetValidationError, "pre-split"):
            parse_dataset_zip(output.getvalue())

    def test_requires_two_classes_and_five_images_each(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for index in range(5):
                archive.writestr(f"cats/{index}.jpg", image_bytes())
            for index in range(4):
                archive.writestr(f"dogs/{index}.jpg", image_bytes())
        with self.assertRaisesRegex(DatasetValidationError, "at least 5"):
            parse_dataset_zip(output.getvalue())

    def test_demo_zip_is_a_valid_200_image_five_class_dataset(self) -> None:
        dataset = parse_dataset_zip(demo_dataset_zip())
        self.assertEqual(dataset["class_counts"], {
            "circle": 40, "diamond": 40, "square": 40, "star": 40, "triangle": 40,
        })


class ImageAndSplitTests(unittest.TestCase):
    def test_image_validation_returns_rgb_and_rejects_limits(self) -> None:
        image = validate_image(image_bytes())
        self.assertEqual((image.mode, image.size), ("RGB", (24, 20)))
        with self.assertRaisesRegex(DatasetValidationError, "5 MB"):
            validate_image(b"x" * (MAX_IMAGE_BYTES + 1))
        with self.assertRaisesRegex(DatasetValidationError, "pixel limit"):
            validate_image(image_bytes(), max_pixels=24 * 20 - 1)

    def test_split_is_deterministic_disjoint_and_stratified(self) -> None:
        dataset = parse_dataset_zip(dataset_zip())
        first = split_dataset(dataset["images"], seed=123)
        second = split_dataset(dataset["images"], seed=123)
        self.assertEqual(first, second)
        self.assertEqual(set().union(*(set(part) for part in first.values())), set(range(12)))
        self.assertEqual(sum(len(part) for part in first.values()), 12)
        for label in dataset["classes"]:
            for indices in first.values():
                self.assertTrue(any(dataset["images"][index]["label"] == label for index in indices))


class MetricsAndDownloadsTests(unittest.TestCase):
    def test_metrics_include_per_class_scores_and_confusion_matrix(self) -> None:
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["cat", "dog"])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertEqual(result["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])

    def test_metrics_reject_mismatched_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            classification_metrics([], [], ["cat", "dog"])
        with self.assertRaisesRegex(ValueError, "unknown class"):
            classification_metrics([0], [3], ["cat", "dog"])

    def test_csv_quotes_values_and_neutralizes_formulas(self) -> None:
        data = rows_to_csv(("class", "filename"), (("=1+1", 'a,"b".jpg'),))
        parsed = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        self.assertEqual(parsed[1], ["'=1+1", 'a,"b".jpg'])
        self.assertEqual(safe_download_stem("../../bad\r\nname.zip"), "bad_name")


if __name__ == "__main__":
    unittest.main()
