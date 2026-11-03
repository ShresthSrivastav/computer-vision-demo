# Vision Studio

Vision Studio is a small, CPU-first computer-vision project workbench. Upload a labeled image dataset, choose a transfer-learning model, train a classifier, and inspect its held-out test results. It is designed for learning and portfolio demos, not for safety-critical or always-on production workloads.

## Workflow

1. Upload a ZIP of labeled images.
2. Choose MobileNetV3-Small or ResNet18, plus a few training settings.
3. Train a new classification head over a frozen ImageNet feature extractor.
4. Review the validation-selected checkpoint's test metrics, learning curves, per-class scores, confusion matrix, and test predictions.
5. Download the state-dictionary model, JSON report, and CSV predictions, or try the model on new images.

This release focuses on **supervised image classification**. Object detection and segmentation require bounding-box or pixel-mask annotations and are not part of this training workflow.

## Dataset format

Put each class of images in a folder named for its label, then zip the folders. An optional outer folder is accepted.

```text
dataset.zip
├── cats/
│   ├── cat_001.jpg
│   └── ...
└── dogs/
    ├── dog_001.jpg
    └── ...
```

JPEG, PNG, and WebP are accepted. Each archive needs 2–15 classes, at least 5 valid images per class, and no more than 250 images total. The upload is limited to 25 MB compressed, 80 MB uncompressed, and 5 MB per image (12 million pixels per image). Do not upload pre-made `train`, `validation`, or `test` directories; the app creates a deterministic, stratified 70/15/15 split so every class appears in each split. Corrupt images are skipped and listed.

The included **Download a tiny example dataset** action creates a synthetic circles-versus-triangles ZIP, so the workflow can be tried without preparing images.

## Models and evaluation

Both choices use TorchVision's ImageNet-1K pretrained weights and train only a new classification head. MobileNetV3-Small is the smaller, faster CPU option; ResNet18 is a larger baseline. The best epoch is chosen using validation loss. The held-out test split is evaluated at the end and reports accuracy, balanced accuracy, macro precision/recall/F1, per-class scores, and a confusion matrix. A small or unrepresentative dataset can produce misleading results; use a larger independent test dataset before relying on a model.

The downloadable `.pt` file contains the model architecture name, class labels, preprocessing metadata, and PyTorch state dictionary. It does not include uploaded images. The JSON report and CSV contain dataset filename, metrics, and test prediction filenames, but no image bytes.

## Run locally

Use Python 3.12. Create a virtual environment and install the pinned dependencies, including CPU-only PyTorch wheels:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Run the automated checks:

```powershell
python -m unittest discover -s tests -v
```

Run one real pretrained CPU training epoch using the generated dataset (downloads MobileNetV3-Small weights on first use):

```powershell
python scripts/smoke_training.py
```

## Deploy on Streamlit Community Cloud

1. Create a Streamlit Community Cloud app from this repository's `main` branch.
2. Set `app.py` as the app entry point.
3. Deploy. Pushes to `main` can trigger a rebuild; the first model run downloads TorchVision weights.

The public app processes uploads in its active session memory and does not write the dataset, trained weights, predictions, or report to persistent storage. Keep uploads modest: free hosting uses shared resources, and CPU training can be slow for larger datasets. Streamlit says resource limits are approximate and may change; apps can hibernate after 12 hours without traffic. This portfolio demo has no uptime guarantee. See [Streamlit Community Cloud resources and hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Model attribution

- **MobileNetV3-Small** and **ResNet18** architectures and ImageNet-1K weights are provided by TorchVision. See [TorchVision models and weight metadata](https://docs.pytorch.org/vision/stable/models.html) and the [TorchVision BSD-3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE).
- The example circles and triangles are generated in memory for this app and do not use external images.

Review upstream model and dataset terms before using trained models beyond learning or prototyping. This app does not guarantee model accuracy or suitability for any particular task.

## Privacy and limits

Dataset ZIPs and prediction images are validated and held in the active Streamlit session while the app runs. The app does not extract archives to disk, save uploads to a database, send images to an inference provider, or log image content. PyTorch's pretrained weights may be downloaded from the official PyTorch host on first training. The repository and app are public; do not upload sensitive images.
