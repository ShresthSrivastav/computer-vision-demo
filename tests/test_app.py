from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class InitialAppFlowTests(unittest.TestCase):
    def test_model_choice_is_visible_before_dataset_upload(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()

        self.assertFalse(app.exception)
        model_picker = next(widget for widget in app.radio if widget.label == "Model architecture")
        self.assertEqual(model_picker.options, ["MobileNetV3-Small", "ResNet18"])
        self.assertEqual(len(app.get("file_uploader")), 1)


if __name__ == "__main__":
    unittest.main()
