import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO

model = YOLO("yolov8l.pt")

def detect(file):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(file)
    results = model.predict(temp.name)
    img = results[0].plot()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="Upload Food Image"),
    outputs=gr.Image(label="Detected Food"),
    title="🍔 Food Calorie Detection using YOLOv8 by Vansh Batra",
    description="Upload food image to detect items and estimate calories"
)

demo.launch()
