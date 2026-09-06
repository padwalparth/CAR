from pathlib import Path
import tempfile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"


st.set_page_config(page_title="Pothole Detector", page_icon="🕳️", layout="wide")
st.title("Pothole Detection Test")
st.caption("Upload an image or video to test the trained YOLOv11 model.")


@st.cache_resource
def load_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    return YOLO(str(WEIGHTS_PATH))


def get_device(model):
    return 0 if getattr(model, "device", None) and model.device.type == "cuda" else "cpu"


def detection_table(result):
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["Class", "Confidence"])

    rows = []
    names = result.names
    for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        rows.append(
            {
                "Class": names[int(class_id)],
                "Confidence": round(float(confidence), 3),
            }
        )
    return pd.DataFrame(rows)


def process_video(model, input_path, output_path, confidence, progress_bar):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError("The annotated video could not be created.")

    frame_number = 0
    total_detections = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(
                source=frame,
                conf=confidence,
                device=get_device(model),
                verbose=False,
            )[0]
            total_detections += len(result.boxes) if result.boxes is not None else 0
            writer.write(result.plot())
            frame_number += 1
            if total_frames:
                progress_bar.progress(min(frame_number / total_frames, 1.0))
    finally:
        capture.release()
        writer.release()

    return frame_number, total_detections


if not WEIGHTS_PATH.exists():
    st.error(f"Could not find the trained model at `{WEIGHTS_PATH}`.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the YOLO model: {error}")
    st.stop()

confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.write(f"Model: `{WEIGHTS_PATH.relative_to(ROOT_DIR)}`")

image_tab, video_tab = st.tabs(["Image", "Video"])

with image_tab:
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )
    if image_file:
        image = Image.open(image_file).convert("RGB")
        result = model.predict(
            source=image,
            conf=confidence,
            device=get_device(model),
            verbose=False,
        )[0]
        annotated_image = result.plot()[:, :, ::-1]
        detections = detection_table(result)

        left, right = st.columns(2)
        with left:
            st.image(annotated_image, caption="Annotated image", use_container_width=True)
        with right:
            st.metric("Detections", len(detections))
            if detections.empty:
                st.info("No potholes detected at this confidence threshold.")
            else:
                st.dataframe(detections, hide_index=True, use_container_width=True)

with video_tab:
    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_uploader",
    )
    if video_file:
        suffix = Path(video_file.name).suffix or ".mp4"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            input_path = temporary_directory / f"input{suffix}"
            output_path = temporary_directory / "potholes_detected.mp4"
            input_path.write_bytes(video_file.getbuffer())

            progress_bar = st.progress(0.0)
            with st.spinner("Processing video frames..."):
                try:
                    frame_count, detection_count = process_video(
                        model, input_path, output_path, confidence, progress_bar
                    )
                except Exception as error:
                    st.error(f"Video processing failed: {error}")
                else:
                    progress_bar.progress(1.0)
                    st.video(output_path.read_bytes())
                    st.write(
                        f"Processed {frame_count:,} frames and found "
                        f"{detection_count:,} detections."
                    )
                    st.download_button(
                        "Download annotated video",
                        data=output_path.read_bytes(),
                        file_name="potholes_detected.mp4",
                        mime="video/mp4",
                    )