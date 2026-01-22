import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🩺 Skin Cancer Detection")
st.write("Upload a skin image to detect cancer type")

# Load YOLO model
model = YOLO("best.pt")

# Upload image
image_file = st.file_uploader(
    "Upload skin image",
    type=["jpg", "jpeg", "png"]
)

if image_file:
    img = cv2.imdecode(
        np.frombuffer(image_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    # YOLO inference
    results = model(img)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        st.image(img, channels="BGR")
        st.success("Healthy Skin")
    else:
        # Take first detected cancer
        box = boxes[0]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        cancer_name = model.names[cls_id]

        output = results[0].plot()

        st.image(output, channels="BGR")
        st.error(f"Cancer Detected: {cancer_name} ({conf:.2f})")

st.markdown("---")
st.caption("⚠️ Academic & research use only. Not for medical diagnosis.")
