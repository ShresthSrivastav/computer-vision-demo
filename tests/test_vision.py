from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from vision import (
    ImageValidationError,
    draw_detection_overlay,
    draw_ocr_overlay,
    draw_segmentation_mask,
    draw_segmentation_overlay,
    image_digest,
    png_bytes,
    rows_to_csv,
    safe_download_stem,
    validate_image,
)


def encoded_image(image_format: str = "JPEG", size: tuple[int, int] = (12, 8)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, (40, 100, 160)).save(stream, format=image_format)
    return stream.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_accepts_and_normalizes_jpeg(self) -> None:
        image = validate_image(encoded_image())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (12, 8))

    def test_accepts_png_and_webp(self) -> None:
        for image_format in ("PNG", "WEBP"):
            with self.subTest(image_format=image_format):
                self.assertEqual(validate_image(encoded_image(image_format)).size, (12, 8))

    def test_rejects_empty_and_corrupt_content(self) -> None:
        for data in (b"", b"not an image"):
            with self.subTest(data=data):
                with self.assertRaises(ImageValidationError):
                    validate_image(data)

    def test_rejects_unsupported_format(self) -> None:
        with self.assertRaisesRegex(ImageValidationError, "JPEG, PNG, or WebP"):
            validate_image(encoded_image("GIF"))

    def test_rejects_byte_and_pixel_limits(self) -> None:
        data = encoded_image("PNG", (20, 20))
        with self.assertRaisesRegex(ImageValidationError, "MB or smaller"):
            validate_image(data, max_bytes=8)
        with self.assertRaisesRegex(ImageValidationError, "too many pixels"):
            validate_image(data, max_pixels=399)

    def test_digest_is_stable_and_content_sensitive(self) -> None:
        first = encoded_image("PNG")
        second = encoded_image("PNG", (13, 8))
        self.assertEqual(image_digest(first), image_digest(first))
        self.assertNotEqual(image_digest(first), image_digest(second))


class ResultRenderingTests(unittest.TestCase):
    def test_detection_overlay_preserves_dimensions_and_png_is_downloadable(self) -> None:
        source = Image.new("RGB", (64, 48), "white")
        overlay = draw_detection_overlay(
            source,
            [{"label": "dog", "confidence": 0.9, "box": [8, 7, 50, 40]}],
        )
        self.assertEqual(overlay.size, source.size)
        self.assertTrue(png_bytes(overlay).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_segmentation_overlay_and_mask_keep_image_size(self) -> None:
        source = Image.new("RGB", (4, 3), "black")
        mask = np.array([[0, 1, 1, 0], [0, 2, 2, 0], [0, 0, 0, 0]], dtype=np.uint8)
        self.assertEqual(draw_segmentation_overlay(source, mask).size, source.size)
        self.assertEqual(draw_segmentation_mask(mask).size, source.size)

    def test_ocr_overlay_and_csv_escape_user_text(self) -> None:
        source = Image.new("RGB", (80, 40), "white")
        overlay = draw_ocr_overlay(
            source,
            [{"text": 'hello, "vision"', "confidence": 0.88, "box": [[4, 5], [70, 5], [70, 30], [4, 30]]}],
        )
        self.assertEqual(overlay.size, source.size)
        csv_data = rows_to_csv(("text", "confidence"), (("hello, \"vision\"", 0.88),))
        self.assertIn(b'"hello, ""vision"""', csv_data)

    def test_csv_prevents_formula_injection_and_download_name_is_safe(self) -> None:
        csv_data = rows_to_csv(("text",), ((" =HYPERLINK(\"https://example.com\")",),))
        self.assertIn(b"' =HYPERLINK", csv_data)
        self.assertEqual(safe_download_stem("../../bad\r\nname.jpg"), "bad_name")


if __name__ == "__main__":
    unittest.main()
