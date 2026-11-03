# Vision Studio

A guided computer-vision lab for exploring how pretrained models interpret images. Try image classification, object detection, semantic segmentation, and OCR on bundled examples or your own photos.

## Try the app

**Deploy the live demo:** In Streamlit Community Cloud, create an app from this repository's `main` branch and set `app.py` as the entrypoint. Add the app URL here after deployment.

The public app processes images in its active session and does not save uploads or results. It runs inference on CPU; the first use of a TorchVision task downloads that model's pretrained weights, so the first run can take longer. OCR model files are bundled with RapidOCR.

## Features

- **Image classification:** rank the top ImageNet categories with confidence scores.
- **Object detection:** draw boxes for detected COCO objects and adjust the confidence threshold.
- **Semantic segmentation:** inspect a colored pixel mask for 20 VOC foreground categories plus background.
- **OCR:** recognize printed text, show text boxes and line confidence, and download TXT or CSV results.
- Download annotated images, masks, prediction tables, or recognized text.
- Use JPEG, PNG, or WebP files up to 10 MB and 12 million pixels.

The models are educational examples. Scores are not guarantees; unsupported objects, handwriting, blur, unusual layouts, and domain-specific images may produce poor results.

## Run locally

Use Python 3.12, then create an environment and install the pinned CPU dependencies:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

Run all four real pretrained models against the bundled samples (downloads TorchVision weights on first use):

```powershell
python scripts/smoke_models.py
```

## Models and attribution

| Task | Model | Labels / training set |
| --- | --- | --- |
| Classification | TorchVision MobileNetV3-Small | ImageNet-1K |
| Detection | TorchVision Faster R-CNN MobileNetV3-Large 320 FPN | COCO |
| Segmentation | TorchVision LRASPP MobileNetV3-Large | COCO images with Pascal VOC labels |
| OCR | RapidOCR PP-OCRv6 small | Printed text; multilingual model |

Torch and TorchVision source code use BSD-3-Clause. Model weights and their training datasets can have separate terms; review the upstream model and dataset terms before reuse beyond this learning demo. RapidOCR is Apache-2.0 and documents its model sources and licenses in its project repository.

- [TorchVision model weights](https://docs.pytorch.org/vision/stable/models.html)
- [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE)
- [RapidOCR installation and bundled models](https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/install_usage/rapidocr/install.en.md)
- [RapidOCR model notices](https://github.com/RapidAI/RapidOCR/blob/main/MODEL_LICENSES.md)
- The Golden Retriever sample is CC0 from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Golden-retriever-dog.jpg); the OCR sample card was generated for this project.

See [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md) for sample-image details.

## Deployment

The app is configured for Streamlit Community Cloud and Python 3.12. Create an app from this public repository with `app.py` as the entrypoint and the `main` branch as its source. Community Cloud can redeploy when commits are pushed to that branch.

Community Cloud is a free portfolio host, not an always-on service. Streamlit documents approximate CPU and memory limits that can change, and apps can hibernate after 12 hours without traffic. First visits after hibernation and first model runs can be slow. See [Community Cloud app resources and limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Privacy

Uploaded images are validated and held in the app's active session for inference. The app does not write image uploads or results to disk, a database, or an external inference provider, and it does not log image content. Pretrained TorchVision weights are downloaded from the official PyTorch host on first use. The app is public; do not upload sensitive images.
