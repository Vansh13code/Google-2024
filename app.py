import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Food Calorie Detector", layout="centered")

st.title("🍔 Food Calorie Detection using YOLO")

# Load model
model = YOLO("yolov8l.pt")   # you already have this file

uploaded_file = st.file_uploader("Upload Food Image or Video", type=["jpg","png","jpeg","mp4"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    if uploaded_file.type.startswith("image"):
        results = model.predict(temp_file.name)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR")

    else:
        st.video(temp_file.name)
