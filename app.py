from __future__ import annotations

import time
from pathlib import Path
from collections import Counter

import gradio as gr
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "FINAL_MODEL" / "steel_defect_yolo11n.pt"

APP_TITLE = "AI Steel Defect Inspection"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640
# ============================================================
# MODEL CHECK AND LOADING
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}\n\n"
        "Make sure this file exists:\n"
        "FINAL_MODEL/steel_defect_yolo11n.pt"
    )

print("=" * 60)
print("AI STEEL DEFECT INSPECTION")
print("=" * 60)
print(f"Model : {MODEL_PATH}")
print(f"Device: {DEVICE}")
print(f"MPS   : {torch.backends.mps.is_available()}")
print("=" * 60)

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Classes:", model.names)
# ============================================================
# MODEL WARM-UP
# ============================================================

def warm_up_model():
    try:
        dummy_image = np.zeros(
            (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
            dtype=np.uint8,
        )

        model.predict(
            source=dummy_image,
            device=DEVICE,
            imgsz=DEFAULT_IMAGE_SIZE,
            conf=DEFAULT_CONFIDENCE,
            verbose=False,
        )

        print("Model warm-up complete.")

    except Exception as exc:
        print(f"Warm-up warning: {exc}")


warm_up_model()
# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_name(class_id: int) -> str:
    names = model.names

    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return f"class_{class_id}"


def normalize_image(image) -> np.ndarray | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.array(image)

    image = np.asarray(image)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    return image.astype(np.uint8)
# ============================================================
# RESULT SUMMARY
# ============================================================

def create_summary(
    detections,
    inference_ms,
    image_shape,
    confidence_threshold,
    iou_threshold,
    image_size,
):
    height, width = image_shape[:2]

    if not detections:
        return f"""
## 🔍 Inspection Result

### No Defects Detected

No defect was detected above the selected confidence threshold.

| Parameter | Value |
|---|---:|
| Image Size | `{width} × {height}` |
| Inference Size | `{image_size}` |
| Confidence Threshold | `{confidence_threshold:.2f}` |
| IoU Threshold | `{iou_threshold:.2f}` |
| Inference Time | `{inference_ms:.1f} ms` |
| Device | `{DEVICE}` |
| Detections | `0` |

> A zero-detection result does not guarantee that the surface is defect-free.
"""

    counts = Counter(
        item["class_name"]
        for item in detections
    )

    count_lines = []

    for class_name, count in sorted(counts.items()):
        count_lines.append(
            f"- **{class_name}**: `{count}`"
        )

    detection_rows = []

    for index, item in enumerate(detections, start=1):
        x1, y1, x2, y2 = item["bbox"]

        detection_rows.append(
            f"| {index} | "
            f"{item['class_name']} | "
            f"{item['confidence'] * 100:.1f}% | "
            f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) |"
        )

    highest_confidence = max(
        item["confidence"]
        for item in detections
    )

    return f"""
## 🔍 Inspection Result

### Defects Detected

**Total detections:** `{len(detections)}`

{chr(10).join(count_lines)}

### Detection Details

| # | Defect | Confidence | Bounding Box |
|---:|---|---:|---|
{chr(10).join(detection_rows)}

### Performance

| Parameter | Value |
|---|---:|
| Image Size | `{width} × {height}` |
| Inference Size | `{image_size}` |
| Inference Time | `{inference_ms:.1f} ms` |
| Device | `{DEVICE}` |
| Confidence Threshold | `{confidence_threshold:.2f}` |
| IoU Threshold | `{iou_threshold:.2f}` |
| Highest Confidence | `{highest_confidence * 100:.1f}%` |
"""
# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_defects(
    image,
    confidence_threshold,
    iou_threshold,
    image_size,
):
    image = normalize_image(image)

    if image is None:
        return (
            None,
            """
## ⚠️ No Image

Please upload a steel surface image.
""",
            [],
        )

    try:
        confidence_threshold = float(confidence_threshold)
        iou_threshold = float(iou_threshold)
        image_size = int(image_size)

        start_time = time.perf_counter()

        results = model.predict(
            source=image,
            device=DEVICE,
            imgsz=image_size,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        inference_ms = (
            time.perf_counter() - start_time
        ) * 1000

        result = results[0]

        detections = []

        if result.boxes is not None:
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                coordinates = (
                    boxes.xyxy[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    float,
                    coordinates,
                )

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": get_class_name(class_id),
                        "confidence": confidence,
                        "bbox": [
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                        ],
                    }
                )

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        annotated_bgr = result.plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2,
        )

        annotated_rgb = (
            annotated_bgr[:, :, ::-1]
            .copy()
        )

        summary = create_summary(
            detections=detections,
            inference_ms=inference_ms,
            image_shape=image.shape,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
        )

        return (
            annotated_rgb,
            summary,
            detections,
        )

    except Exception as exc:
        error_message = (
            f"## ❌ Detection Error\n\n"
            f"**{type(exc).__name__}:**\n\n"
            f"{exc}"
        )

        return (
            None,
            error_message,
            [],
        )
# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_all():
    return (
        None,
        None,
        """
## 🟢 Inspection Ready

Upload a steel surface image and click **Inspect Surface**.
""",
        [],
    )
# ============================================================
# UI STYLING
# ============================================================

CSS = """
body {
    background: #0b0f14;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
}

.hero {
    text-align: center;
    padding: 25px 10px 20px 10px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    opacity: 0.75;
    font-size: 16px;
}

footer {
    display: none !important;
}
"""
# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title=APP_TITLE,
    css=CSS,
    theme=gr.themes.Soft(),
) as app:

    gr.HTML(
        """
        <div class="hero">
            <h1>🔬 AI Steel Defect Inspection</h1>
            <p>
                YOLO-based Computer Vision System for Automated
                Steel Surface Defect Detection and Localization
            </p>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            input_image = gr.Image(
                type="numpy",
                label="📤 Input Steel Surface",
            )

            with gr.Accordion(
                "⚙️ Detection Settings",
                open=True,
            ):

                confidence_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=DEFAULT_CONFIDENCE,
                    step=0.01,
                    label="Confidence Threshold",
                    info=(
                        "Lower values can detect weaker defects "
                        "but may increase false positives."
                    ),
                )

                iou_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=DEFAULT_IOU,
                    step=0.01,
                    label="IoU / NMS Threshold",
                )

                image_size_dropdown = gr.Dropdown(
                    choices=[
                        320,
                        416,
                        512,
                        640,
                        768,
                        960,
                        1280,
                    ],
                    value=DEFAULT_IMAGE_SIZE,
                    label="Inference Image Size",
                )

            with gr.Row():

                detect_button = gr.Button(
                    "🔍 Inspect Surface",
                    variant="primary",
                    size="lg",
                )

                clear_button = gr.Button(
                    "↻ Clear",
                    variant="secondary",
                    size="lg",
                )

        with gr.Column(scale=1):

            output_image = gr.Image(
                label="🎯 Detection Result",
                type="numpy",
            )

            result_markdown = gr.Markdown(
                """
## 🟢 Inspection Ready

Upload a steel surface image and click **Inspect Surface**.
"""
            )
            # ============================================================
# DETECTION DATA + MODEL INFORMATION
# ============================================================

    with gr.Accordion(
        "📋 Detection Data",
        open=False,
    ):

        detection_json = gr.JSON(
    label="Detection Data",
    value=[],
)

    gr.Markdown(
        f"""
---

## 🧠 Model Information

| Component | Configuration |
|---|---|
| Model | `YOLO11n` |
| Dataset | `NEU-DET` |
| Task | Steel Surface Defect Detection |
| Model File | `{MODEL_PATH.name}` |
| Runtime Device | `{DEVICE}` |
| MPS Available | `{torch.backends.mps.is_available()}` |
| Default Confidence | `{DEFAULT_CONFIDENCE}` |
| Default IoU | `{DEFAULT_IOU}` |
"""
    )
    # ============================================================
# BUTTON ACTIONS
# ============================================================

    detect_button.click(
        fn=detect_defects,
        inputs=[
            input_image,
            confidence_slider,
            iou_slider,
            image_size_dropdown,
        ],
        outputs=[
            output_image,
            result_markdown,
            detection_json,
        ],
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            input_image,
            output_image,
            result_markdown,
            detection_json,
        ],
    )
    # ============================================================
# APPLICATION LAUNCH
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("STARTING AI STEEL DEFECT INSPECTION UI")
    print("=" * 60)
    print(f"Model : {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print("Opening browser...")
    print("=" * 60)

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
show_api=False,
    )
    