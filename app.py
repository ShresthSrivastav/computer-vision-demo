from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from vision import (
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    TASK_OCR,
    TASK_SEGMENTATION,
    ImageValidationError,
    build_model,
    classify,
    detect,
    draw_detection_overlay,
    draw_ocr_overlay,
    draw_segmentation_mask,
    draw_segmentation_overlay,
    image_digest,
    png_bytes,
    recognize_text,
    rows_to_csv,
    safe_download_stem,
    segment,
    validate_image,
)

LOGGER = logging.getLogger("vision_studio")
APP_DIR = Path(__file__).parent
ASSET_DIR = APP_DIR / "assets"
RESULT_KEY = "vision_studio_result"
UPLOAD_VERSION_KEY = "vision_studio_upload_version"

TASK_DESCRIPTIONS = {
    TASK_CLASSIFICATION: (
        "What is most likely in this image? The model ranks ImageNet classes;"
        " these scores are relative model confidence, not calibrated probabilities."
    ),
    TASK_DETECTION: (
        "Where are familiar objects? Adjust the confidence threshold to filter boxes."
        " The detector recognizes the 80 COCO object classes."
    ),
    TASK_SEGMENTATION: (
        "Which pixels belong to each scene category? This lightweight model predicts"
        " 20 foreground VOC categories plus background."
    ),
    TASK_OCR: (
        "Find and read visible text. The bundled RapidOCR model is tuned for printed"
        " text; results can be weaker on handwriting, blur, or unusual layouts."
    ),
}


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_model(task: str) -> Any:
    return build_model(task)


def clear_session_upload() -> None:
    version = int(st.session_state.get(UPLOAD_VERSION_KEY, 0))
    st.session_state.pop(f"vision_upload_{version}", None)
    st.session_state.pop(RESULT_KEY, None)
    st.session_state[UPLOAD_VERSION_KEY] = version + 1


def result_signature(task: str, raw: bytes, settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(task.encode() + raw + payload).hexdigest()


def load_source_image() -> tuple[Image.Image | None, bytes, str, str]:
    st.sidebar.markdown("### Image source")
    source = st.sidebar.radio(
        "Choose an image",
        ("Sample gallery", "Upload an image"),
        label_visibility="collapsed",
    )

    if source == "Sample gallery":
        samples = {
            "Golden retriever": (ASSET_DIR / "golden_retriever.jpg", "A public-domain dog photograph."),
            "Printed text card": (ASSET_DIR / "ocr_sample.png", "A high-contrast OCR sample generated for this project."),
        }
        choice = st.sidebar.selectbox("Sample image", tuple(samples))
        path, description = samples[choice]
        raw = path.read_bytes()
        name = path.name
    else:
        version = int(st.session_state.get(UPLOAD_VERSION_KEY, 0))
        uploaded = st.sidebar.file_uploader(
            "Choose a JPEG, PNG, or WebP",
            type=("jpg", "jpeg", "png", "webp"),
            max_upload_size=10,
            key=f"vision_upload_{version}",
            help="Up to 10 MB. Images are processed in memory and are not saved.",
        )
        if uploaded is None:
            return None, b"", "", "Choose an image to begin."
        raw = uploaded.getvalue()
        name = uploaded.name
        description = "Your upload is held in this app session only."

    try:
        image = validate_image(raw)
    except ImageValidationError as exc:
        return None, raw, name, str(exc)
    return image, raw, name, description


def run_task(task: str, bundle: Any, image: Image.Image, settings: dict[str, Any]) -> dict[str, Any]:
    if task == TASK_CLASSIFICATION:
        return {"predictions": classify(bundle, image, settings["top_k"])}
    if task == TASK_DETECTION:
        detections = detect(bundle, image, settings["threshold"])
        return {"detections": detections, "overlay": draw_detection_overlay(image, detections)}
    if task == TASK_SEGMENTATION:
        mask, regions = segment(bundle, image)
        return {
            "regions": regions,
            "overlay": draw_segmentation_overlay(image, mask),
            "mask_image": draw_segmentation_mask(mask),
        }
    lines = recognize_text(bundle, image)
    return {"lines": lines, "overlay": draw_ocr_overlay(image, lines)}


def render_classification(result: dict[str, Any], filename: str) -> None:
    predictions = result["predictions"]
    best = predictions[0]
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Ranked predictions")
        for row in predictions:
            st.markdown(f"**{row['rank']:02d} · {row['label']}** · {row['confidence']:.1%}")
            st.progress(row["confidence"])
    with right:
        st.markdown("#### Leading class")
        st.metric(best["label"], f"{best['confidence']:.1%}")
        st.caption("Scores rank this model’s alternatives. They are not a guarantee that the image contains the class.")
        st.download_button(
            "Download predictions · CSV",
            rows_to_csv(("rank", "class", "confidence"), ((r["rank"], r["label"], r["confidence"]) for r in predictions)),
            file_name=f"{Path(filename).stem}_classification.csv",
            mime="text/csv",
            key="download_classification",
            width="stretch",
        )


def render_detection(result: dict[str, Any], filename: str) -> None:
    detections = result["detections"]
    if not detections:
        st.info("No objects were found above this confidence threshold. Try lowering it or choosing another image.")
    else:
        st.markdown(f"#### {len(detections)} object{'' if len(detections) == 1 else 's'} found")
    st.image(result["overlay"], caption="Predictions from the COCO-trained detector", width="stretch")
    if detections:
        st.dataframe(
            [
                {
                    "Object": item["label"],
                    "Confidence": f"{item['confidence']:.1%}",
                    "Box (left, top, right, bottom)": ", ".join(f"{v:.0f}" for v in item["box"]),
                }
                for item in detections
            ],
            hide_index=True,
            width="stretch",
        )
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download annotated image · PNG",
            png_bytes(result["overlay"]),
            file_name=f"{Path(filename).stem}_detections.png",
            mime="image/png",
            key="download_detections_image",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download detections · CSV",
            rows_to_csv(
                ("class", "confidence", "left", "top", "right", "bottom"),
                ((d["label"], d["confidence"], *[round(v, 2) for v in d["box"]]) for d in detections),
            ),
            file_name=f"{Path(filename).stem}_detections.csv",
            mime="text/csv",
            key="download_detections_csv",
            width="stretch",
        )


def render_segmentation(result: dict[str, Any], filename: str) -> None:
    left, right = st.columns(2)
    with left:
        st.image(result["overlay"], caption="Semantic regions overlaid on the source image", width="stretch")
    with right:
        st.image(result["mask_image"], caption="Color mask · background is transparent in the overlay", width="stretch")
    regions = result["regions"]
    if not regions:
        st.info("The model assigned all pixels to background for this image.")
    else:
        st.markdown("#### Largest regions")
        st.dataframe(
            [
                {"Category": row["label"], "Image area": f"{row['share']:.1%}", "Pixels": row["pixels"]}
                for row in regions[:10]
            ],
            hide_index=True,
            width="stretch",
        )
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download overlay · PNG",
            png_bytes(result["overlay"]),
            file_name=f"{Path(filename).stem}_segmentation_overlay.png",
            mime="image/png",
            key="download_segmentation_overlay",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download class mask · PNG",
            png_bytes(result["mask_image"]),
            file_name=f"{Path(filename).stem}_segmentation_mask.png",
            mime="image/png",
            key="download_segmentation_mask",
            width="stretch",
        )


def render_ocr(result: dict[str, Any], filename: str) -> None:
    lines = result["lines"]
    st.image(result["overlay"], caption="Detected text regions and recognition confidence", width="stretch")
    if not lines:
        st.info("No readable text was found. Try a sharper image with larger, well-lit printed text.")
        return
    st.markdown("#### Recognized text")
    st.code("\n".join(line["text"] for line in lines), language=None, wrap_lines=True)
    st.dataframe(
        [{"Text": line["text"], "Confidence": f"{line['confidence']:.1%}"} for line in lines],
        hide_index=True,
        width="stretch",
    )
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download recognized text · TXT",
            "\n".join(line["text"] for line in lines).encode("utf-8"),
            file_name=f"{Path(filename).stem}_text.txt",
            mime="text/plain; charset=utf-8",
            key="download_ocr_text",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download OCR boxes · CSV",
            rows_to_csv(
                ("text", "confidence", "box_points"),
                ((line["text"], line["confidence"], json.dumps(line["box"])) for line in lines),
            ),
            file_name=f"{Path(filename).stem}_ocr.csv",
            mime="text/csv",
            key="download_ocr_csv",
            width="stretch",
        )


def render_result(task: str, result: dict[str, Any], filename: str) -> None:
    if task == TASK_CLASSIFICATION:
        render_classification(result, filename)
    elif task == TASK_DETECTION:
        render_detection(result, filename)
    elif task == TASK_SEGMENTATION:
        render_segmentation(result, filename)
    else:
        render_ocr(result, filename)


def main() -> None:
    st.set_page_config(
        page_title="Vision Studio",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(148, 163, 184, .14); }
        .hero-kicker { color: #9de6c3; font-size: .78rem; font-weight: 750; letter-spacing: .16em; text-transform: uppercase; }
        .hero-title { color: #f8fafc; font-size: clamp(2.2rem, 5vw, 3.8rem); font-weight: 780; letter-spacing: -.055em; line-height: 1.02; margin: .4rem 0 .8rem; }
        .hero-copy { color: #afbdd0; font-size: 1.08rem; max-width: 720px; line-height: 1.65; }
        .status-chip { display: inline-block; margin: .8rem .45rem 0 0; padding: .38rem .7rem; border: 1px solid rgba(157,230,195,.2); border-radius: 99px; color: #c9f5df; background: rgba(157,230,195,.07); font-size: .78rem; }
        div[data-testid="stMetric"] { background: rgba(18,29,46,.8); border: 1px solid rgba(148,163,184,.14); padding: .9rem 1rem; border-radius: 14px; }
        div[data-testid="stMetricLabel"] { color: #9aabc0; }
        div[data-testid="stButton"] > button[kind="primary"] { border: 0; color: #07111f; font-weight: 750; }
        div[data-testid="stAlert"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if UPLOAD_VERSION_KEY not in st.session_state:
        st.session_state[UPLOAD_VERSION_KEY] = 0

    with st.sidebar:
        st.markdown("## 🔎 Vision Studio")
        st.caption("A guided lab for seeing how computer-vision models interpret images.")
        st.button("Clear current image and result", on_click=clear_session_upload, width="stretch")
        st.divider()
        st.markdown("### Privacy and runtime")
        st.info("Images are processed in this app session and are not saved. Models run on CPU; first use may download model weights.")
        st.caption("Streamlit Community Cloud can sleep when idle and has shared resource limits.")
        st.markdown("[Source and setup](https://github.com/ShresthSrivastav/computer-vision-demo)")

    st.markdown('<div class="hero-kicker">A hands-on computer-vision lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Teach a machine to see.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Explore image classification, object detection, semantic segmentation, and OCR with compact pretrained models. Try the samples, then bring an image of your own.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="status-chip">CPU inference</span><span class="status-chip">Open pretrained models</span><span class="status-chip">Session-only images</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    task = st.radio(
        "Choose a vision lab",
        (TASK_CLASSIFICATION, TASK_DETECTION, TASK_SEGMENTATION, TASK_OCR),
        horizontal=True,
        label_visibility="collapsed",
        key="vision_task",
    )
    st.markdown(f"### {task}")
    st.write(TASK_DESCRIPTIONS[task])

    image, raw, filename, source_note = load_source_image()
    settings: dict[str, Any] = {}
    if task == TASK_CLASSIFICATION:
        settings["top_k"] = st.select_slider("Show top predictions", options=(3, 5, 10), value=5)
    elif task == TASK_DETECTION:
        settings["threshold"] = st.slider("Minimum confidence", 0.10, 0.90, 0.55, 0.05, format="%.2f")

    if image is None:
        st.session_state.pop(RESULT_KEY, None)
        if source_note and source_note != "Choose an image to begin.":
            st.error(source_note)
        else:
            st.info(source_note)
        return

    signature = result_signature(task, raw, settings)
    current_result = st.session_state.get(RESULT_KEY)
    if current_result and current_result.get("signature") != signature:
        st.session_state.pop(RESULT_KEY, None)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.image(image, caption=f"{filename} · {image.width:,} × {image.height:,}", width="stretch")
        st.caption(source_note)
    with right:
        st.markdown("#### What this model does")
        st.write(
            {
                TASK_CLASSIFICATION: "Rank likely labels for the full image.",
                TASK_DETECTION: "Locate objects and show their confidence and coordinates.",
                TASK_SEGMENTATION: "Color pixels by the scene category predicted for each region.",
                TASK_OCR: "Read printed text and return its location and confidence.",
            }[task]
        )
        st.markdown("#### Model notes")
        if task == TASK_CLASSIFICATION:
            st.caption("MobileNetV3-Small · ImageNet-1K · 1,000 categories · optimized for mobile inference.")
        elif task == TASK_DETECTION:
            st.caption("Faster R-CNN with MobileNetV3-Large · COCO · 80 object categories · low-resolution variant for CPU use.")
        elif task == TASK_SEGMENTATION:
            st.caption("LRASPP with MobileNetV3-Large · COCO/VOC labels · 20 foreground categories plus background.")
        else:
            st.caption("RapidOCR PP-OCRv6 small · printed text recognition · Latin and CJK text may be detected.")
        run = st.button("Run analysis", type="primary", width="stretch")

    if run:
        started = time.perf_counter()
        try:
            with st.spinner("Loading the model and analyzing this image on CPU…"):
                bundle = cached_model(task)
                result = run_task(task, bundle, image, settings)
            st.session_state[RESULT_KEY] = {
                "signature": signature,
                "task": task,
                "filename": filename,
                "result": result,
                "elapsed": time.perf_counter() - started,
            }
        except Exception as exc:  # noqa: BLE001 - keep user-facing errors clear and log diagnostics.
            LOGGER.exception("Inference failed for task %s", task)
            st.error("This model could not finish the analysis. Check the app logs, then retry; model weights may still be downloading on first use.")
            st.caption(f"Error type: {type(exc).__name__}")

    saved = st.session_state.get(RESULT_KEY)
    if saved and saved.get("signature") == signature:
        st.divider()
        left, right = st.columns([1, 5])
        with left:
            st.metric("Run time", f"{saved['elapsed']:.2f}s")
        with right:
            st.caption("Total time includes loading a model the first time it is used in this app session.")
        render_result(task, saved["result"], safe_download_stem(filename))
    else:
        st.divider()
        st.markdown("#### Your result will appear here")
        st.caption("Choose a sample or upload an image, then run the model. Only the selected task is loaded.")

    with st.expander("Learning notes and limitations"):
        st.markdown(
            """
            - **Classification** ranks one whole image against ImageNet labels. It does not explain where an object appears.
            - **Detection** returns boxes for familiar COCO classes. Small, overlapping, or unusual objects may be missed.
            - **Segmentation** assigns scene pixels to a compact VOC label set. The overlay is a model prediction, not a measured ground truth.
            - **OCR** reads printed text and shows box-level scores. Handwriting, perspective, glare, and low resolution can reduce accuracy.

            Confidence is a model score, not a guarantee. These examples are for learning and are not appropriate for safety-critical decisions.
            """
        )


if __name__ == "__main__":
    main()
