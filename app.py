from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from vision import (
    ARCHITECTURES,
    DatasetValidationError,
    demo_dataset_zip,
    json_bytes,
    model_artifact_bytes,
    parse_dataset_zip,
    predict_images,
    rows_to_csv,
    safe_download_stem,
    split_dataset,
    train_classifier,
)

LOGGER = logging.getLogger("vision_studio")
DATASET_KEY = "vision_studio_dataset"
ERROR_KEY = "vision_studio_dataset_error"
RESULT_KEY = "vision_studio_training_result"
UPLOAD_VERSION_KEY = "vision_studio_upload_version"


@st.cache_data(max_entries=1, show_spinner=False)
def example_dataset_zip() -> bytes:
    return demo_dataset_zip()


def clear_workspace() -> None:
    version = int(st.session_state.get(UPLOAD_VERSION_KEY, 0)) + 1
    for key in (
        DATASET_KEY, ERROR_KEY, RESULT_KEY, "vision_studio_dataset_signature", "vision_studio_dataset_name",
        "vision_studio_training_signature", "vision_studio_training_settings",
        "vision_studio_new_prediction_signature", "vision_studio_new_predictions",
    ):
        st.session_state.pop(key, None)
    st.session_state[UPLOAD_VERSION_KEY] = version


def set_dataset(raw: bytes, filename: str) -> None:
    signature = hashlib.sha256(raw).hexdigest()
    if signature == st.session_state.get("vision_studio_dataset_signature"):
        st.session_state["vision_studio_dataset_name"] = filename
        return
    st.session_state.pop(RESULT_KEY, None)
    st.session_state["vision_studio_dataset_signature"] = signature
    st.session_state["vision_studio_dataset_name"] = filename
    try:
        st.session_state[DATASET_KEY] = parse_dataset_zip(raw)
        st.session_state.pop(ERROR_KEY, None)
    except DatasetValidationError as exc:
        st.session_state.pop(DATASET_KEY, None)
        st.session_state[ERROR_KEY] = str(exc)


def render_results(run: dict[str, Any], settings: dict[str, Any], dataset_name: str) -> None:
    metrics = run["metrics"]
    st.divider()
    st.markdown("## Evaluation results")
    st.caption(
        f"Best checkpoint selected by validation loss · measured once on the held-out test split · "
        f"{run['test_size']} test images · {run['elapsed']:.1f}s total training time"
    )
    cards = st.columns(4)
    for column, label, value, help_text in zip(
        cards,
        ("Test accuracy", "Balanced accuracy", "Macro F1", "Test loss"),
        (metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"], run["test_loss"]),
        (
            "Share of all test predictions that are correct.",
            "Average recall across classes; useful when classes have different sizes.",
            "Average class-level F1 score, treating each class equally.",
            "Cross-entropy on the held-out test images.",
        ),
    ):
        column.metric(label, f"{value:.1%}" if label != "Test loss" else f"{value:.3f}", help=help_text)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Learning curves")
        history = pd.DataFrame(run["history"]).set_index("epoch")
        st.line_chart(history[["train_accuracy", "validation_accuracy"]], y_label="Accuracy", x_label="Epoch")
        st.line_chart(history[["train_loss", "validation_loss"]], y_label="Loss", x_label="Epoch")
    with right:
        st.markdown("### Confusion matrix")
        matrix = pd.DataFrame(run["metrics"]["confusion_matrix"], index=run["labels"], columns=run["labels"])
        matrix.index.name = "Actual ↓  /  Predicted →"
        st.dataframe(matrix, width="stretch")
        st.caption("Rows are actual classes; columns are model predictions.")

    st.markdown("### Per-class test scores")
    class_frame = pd.DataFrame(run["metrics"]["per_class"]).rename(
        columns={"class": "Class", "precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Test images"}
    )
    st.dataframe(class_frame, hide_index=True, width="stretch", column_config={
        "Precision": st.column_config.NumberColumn(format="%.1%%"),
        "Recall": st.column_config.NumberColumn(format="%.1%%"),
        "F1": st.column_config.NumberColumn(format="%.1%%"),
    })
    predictions = pd.DataFrame(run["predictions"]).rename(columns={
        "filename": "Image", "actual": "Actual class", "predicted": "Predicted class", "confidence": "Confidence", "correct": "Correct"
    })
    predictions["Confidence"] = predictions["Confidence"].map(lambda score: f"{score:.1%}")
    st.markdown("### Test-set predictions")
    st.dataframe(predictions, hide_index=True, width="stretch")

    report = {
        "dataset": {"filename": dataset_name, "classes": run["labels"], "split_seed": settings["seed"]},
        "training": settings,
        "model": {"architecture": run["architecture"], "preprocessing": run["preprocessing"]},
        "test_size": run["test_size"],
        "test_loss": run["test_loss"],
        "metrics": run["metrics"],
        "history": run["history"],
        "predictions": run["predictions"],
    }
    stem = safe_download_stem(Path(dataset_name).stem)
    st.markdown("### Take your run with you")
    downloads = st.columns(3)
    with downloads[0]:
        st.download_button("Download model weights", model_artifact_bytes(run), file_name=f"{stem}_{run['architecture'].lower()}_model.pt", mime="application/octet-stream", width="stretch", help="State dictionary, class labels, and preprocessing metadata.")
    with downloads[1]:
        st.download_button("Download run report", json_bytes(report), file_name=f"{stem}_report.json", mime="application/json", width="stretch")
    with downloads[2]:
        st.download_button(
            "Download test predictions",
            rows_to_csv(
                ("filename", "actual_class", "predicted_class", "confidence", "correct"),
                ((p["filename"], p["actual"], p["predicted"], p["confidence"], p["correct"]) for p in run["predictions"]),
            ),
            file_name=f"{stem}_test_predictions.csv", mime="text/csv", width="stretch",
        )

    st.markdown("### Try your trained model")
    new_images = st.file_uploader(
        "Upload new images to classify",
        type=("jpg", "jpeg", "png", "webp"),
        accept_multiple_files=True,
        max_upload_size=5,
        key=f"vision_new_images_{st.session_state.get(UPLOAD_VERSION_KEY, 0)}",
        help="Up to 20 images, 5 MB each. Images are processed in session memory.",
    )
    uploads = [(file.name, file.getvalue()) for file in (new_images or [])]
    prediction_payload = run.get("run_signature", "") + "\0" + "".join(name for name, _ in uploads)
    prediction_signature = hashlib.sha256(prediction_payload.encode() + b"\0" + b"".join(data for _, data in uploads)).hexdigest()
    saved_signature = st.session_state.get("vision_studio_new_prediction_signature")
    if prediction_signature != saved_signature:
        st.session_state.pop("vision_studio_new_predictions", None)
    if len(uploads) > 20:
        st.error("Choose no more than 20 images at a time.")
    elif uploads and st.button("Classify new images", key=f"vision_predict_{st.session_state.get(UPLOAD_VERSION_KEY, 0)}"):
        try:
            with st.spinner("Classifying with your trained model…"):
                new_predictions = predict_images(run, uploads)
            st.session_state["vision_studio_new_predictions"] = new_predictions
            st.session_state["vision_studio_new_prediction_signature"] = prediction_signature
        except DatasetValidationError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - keep inference errors out of the uploaded-image UI.
            LOGGER.exception("Prediction failed for the trained model")
            st.error("The new images could not be classified. Check the app logs and try again.")
            st.caption(f"Error type: {type(exc).__name__}")
    new_predictions = st.session_state.get("vision_studio_new_predictions", [])
    if new_predictions and st.session_state.get("vision_studio_new_prediction_signature") == prediction_signature:
        st.dataframe(
            [{"Image": row["filename"], "Predicted class": row["predicted"], "Confidence": f"{row['confidence']:.1%}"} for row in new_predictions],
            hide_index=True,
            width="stretch",
        )


def main() -> None:
    st.set_page_config(page_title="Vision Studio · Train", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .block-container { max-width: 1280px; padding-top: 2.1rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(148,163,184,.14); }
        .eyebrow { color:#9de6c3; font-size:.76rem; font-weight:750; letter-spacing:.16em; text-transform:uppercase; }
        .hero { color:#f8fafc; font-size:clamp(2.25rem,5vw,3.9rem); font-weight:790; letter-spacing:-.055em; line-height:1.02; margin:.45rem 0 .75rem; }
        .hero-copy { color:#afbdd0; font-size:1.06rem; max-width:780px; line-height:1.65; }
        .step-chip { display:inline-block; margin:.9rem .4rem 0 0; padding:.36rem .68rem; border:1px solid rgba(157,230,195,.22); border-radius:99px; color:#c9f5df; background:rgba(157,230,195,.07); font-size:.78rem; }
        div[data-testid="stMetric"] { background:rgba(18,29,46,.8); border:1px solid rgba(148,163,184,.14); padding:.85rem 1rem; border-radius:14px; }
        div[data-testid="stMetricLabel"] { color:#9aabc0; }
        div[data-testid="stButton"] > button[kind="primary"] { border:0; color:#07111f; font-weight:750; }
        div[data-testid="stAlert"] { border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.setdefault(UPLOAD_VERSION_KEY, 0)
    with st.sidebar:
        st.markdown("## 👁️ Vision Studio")
        st.caption("A guided workbench for training image classifiers on your own labeled data.")
        st.button("Clear dataset and results", on_click=clear_workspace, width="stretch")
        st.divider()
        st.markdown("### Dataset format")
        st.code("dataset.zip\n├── cats/\n│   ├── cat_01.jpg\n│   └── ...\n└── dogs/\n    ├── dog_01.jpg\n    └── ...", language=None)
        st.caption("At least 5 valid images per class · 2–15 classes · no fixed image-count cap")
        st.divider()
        st.markdown("### Privacy and runtime")
        st.info("Your image data stays in this app session's memory. The app does not save uploads or results.")
        st.caption("CPU training only. Pretrained ImageNet weights download on the first run. Cloud resources are shared and may sleep when idle.")
        st.markdown("[Source code and setup](https://github.com/ShresthSrivastav/computer-vision-demo)")

    st.markdown('<div class="eyebrow">Your dataset · Your model · Your evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero">Train a vision model.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Upload labeled images, choose a pretrained architecture, and train a custom classifier with transfer learning. Vision Studio creates the data split, tracks learning, and evaluates on images the model never trained on.</div>', unsafe_allow_html=True)
    st.markdown('<span class="step-chip">01 · Upload data</span><span class="step-chip">02 · Choose a model</span><span class="step-chip">03 · Train & evaluate</span><span class="step-chip">CPU only</span>', unsafe_allow_html=True)
    st.write("")

    st.markdown("## 1 · Add your image dataset")
    st.write("Upload a ZIP with one folder per class. An optional outer folder is fine. Images can be JPEG, PNG, or WebP.")
    version = st.session_state[UPLOAD_VERSION_KEY]
    uploaded = st.file_uploader(
        "Dataset ZIP",
        type=("zip",),
        max_upload_size=50,
        key=f"vision_dataset_upload_{version}",
        help="Maximum 50 MB compressed and 80 MB uncompressed. No fixed image-count cap; a 10,000-entry archive safety cap applies. Images stay in session memory.",
    )
    if uploaded is not None:
        set_dataset(uploaded.getvalue(), uploaded.name)
    elif DATASET_KEY not in st.session_state:
        st.download_button(
            "Download the 200-image example dataset",
            example_dataset_zip(),
            file_name="vision_studio_shapes.zip",
            mime="application/zip",
            help="A balanced synthetic dataset with 5 shape classes and 40 images per class. It fits the upload limits.",
        )

    st.markdown("## 2 · Choose your model")
    architecture = st.radio(
        "Model architecture",
        ARCHITECTURES,
        horizontal=True,
        key="vision_architecture",
        help="Choose a pretrained image feature extractor. Vision Studio trains a new classification head for your labels.",
    )
    st.caption("MobileNetV3-Small is lighter and faster on CPU. ResNet18 is a larger classic baseline.")

    if ERROR_KEY in st.session_state:
        st.error(st.session_state[ERROR_KEY])
    dataset = st.session_state.get(DATASET_KEY)
    if not dataset:
        st.info("Your model is selected. Upload a valid ZIP above to see the class balance, configure training, and start the run.")
        st.stop()

    counts = dataset["class_counts"]
    image_count = len(dataset["images"])
    split = split_dataset(dataset["images"])
    current_signature = st.session_state.get("vision_studio_dataset_signature")

    summary = st.columns(3)
    summary[0].metric("Images ready", f"{image_count:,}")
    summary[1].metric("Classes", len(counts))
    summary[2].metric("Train / validation / test", f"{len(split['train'])} / {len(split['validation'])} / {len(split['test'])}")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("### Images per class")
        st.bar_chart(pd.Series(counts, name="Images"), y_label="Images")
    with right:
        st.markdown("### Automatic stratified split")
        st.dataframe(
            [{"Split": name.title(), "Images": len(split[name]), "Share": f"{len(split[name]) / image_count:.0%}"} for name in ("train", "validation", "test")],
            hide_index=True,
            width="stretch",
        )
        st.caption("Every class is represented in all three splits. Test images remain held out until final evaluation.")
    if dataset["warnings"]:
        with st.expander(f"Skipped files ({len(dataset['warnings'])}{'+' if len(dataset['warnings']) == 3 else ''})"):
            for warning in dataset["warnings"]:
                st.caption(warning)

    st.divider()
    st.markdown("## 3 · Configure training")
    model_columns = st.columns([1.1, .8, 1])
    with model_columns[0]:
        epochs = st.slider("Epochs", 1, 6, 3, help="An epoch is one pass through the training split. The best checkpoint is selected by validation loss.")
    with model_columns[1]:
        batch_size = st.selectbox("Batch size", (4, 8, 16), index=1)
    with model_columns[2]:
        learning_rate = st.select_slider("Learning rate", options=(0.0001, 0.0003, 0.001, 0.003, 0.01), value=0.001, format_func=lambda value: f"{value:g}")
    st.caption(
        "Both models start from TorchVision ImageNet weights. The pretrained feature extractor stays frozen; only the new class head is trained."
    )
    settings = {"architecture": architecture, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": 42}
    run_signature = hashlib.sha256(json.dumps({"dataset": current_signature, "settings": settings}, sort_keys=True).encode()).hexdigest()
    train_clicked = st.button("Train model", type="primary", width="stretch", key="train_vision_model")
    if train_clicked:
        st.session_state.pop(RESULT_KEY, None)
        progress = st.progress(0, text="Preparing the model and dataset…")
        status = st.empty()
        started = time.perf_counter()

        def on_epoch(epoch: int, total: int, row: dict[str, float]) -> None:
            progress.progress(epoch / total, text=f"Training · epoch {epoch} of {total}")
            status.caption(
                f"Train accuracy {row['train_accuracy']:.1%} · validation accuracy {row['validation_accuracy']:.1%} · "
                f"validation loss {row['validation_loss']:.3f}"
            )

        try:
            with st.spinner("Loading ImageNet weights if needed, then training on CPU…"):
                run = train_classifier(
                    dataset["images"], split, architecture,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, on_epoch=on_epoch,
                )
            run["elapsed"] = time.perf_counter() - started
            run["test_size"] = len(split["test"])
            run["run_signature"] = run_signature
            st.session_state[RESULT_KEY] = run
            st.session_state["vision_studio_training_signature"] = run_signature
            st.session_state["vision_studio_training_settings"] = settings
            progress.progress(1.0, text="Training and test evaluation complete")
            status.empty()
        except Exception as exc:  # noqa: BLE001 - log details without exposing uploaded content.
            LOGGER.exception("Training failed for %s", architecture)
            st.error("Training could not complete. Check the app logs; the first run may need time to download model weights.")
            st.caption(f"Error type: {type(exc).__name__}")

    run = st.session_state.get(RESULT_KEY)
    if run and st.session_state.get("vision_studio_training_signature") == run_signature:
        render_results(run, st.session_state["vision_studio_training_settings"], st.session_state["vision_studio_dataset_name"])
    else:
        st.divider()
        st.markdown("## 4 · Train, inspect, and export")
        if run:
            st.info("The model settings changed. Train again to see results for this configuration.")
        else:
            st.caption("Your evaluation metrics, learning curves, confusion matrix, predictions, and downloadable model appear here after training.")

    with st.expander("How to read the results"):
        st.markdown(
            """
            - **Training split** updates the new classifier head. **Validation split** selects the best epoch. **Test split** is used once for final metrics.
            - **Accuracy** is the fraction correct. **Balanced accuracy** and **macro F1** give each class equal weight, which helps reveal weak classes in an imbalanced dataset.
            - The displayed class confidence is a softmax score, not a calibrated probability or a guarantee.
            - These small CPU runs are for learning and prototyping. A tiny dataset, label noise, or a domain unlike ImageNet can produce misleadingly high or low scores. Use a larger independent dataset before making real-world decisions.
            - The exported `.pt` file contains a PyTorch state dictionary, class labels, and preprocessing metadata. It does not contain the uploaded images.
            """
        )


if __name__ == "__main__":
    main()
