import cv2
from ultralytics import YOLO
model=YOLO("yolov8l.pt")
results=model("Image1.jpg",show=True)
cv2.waitKey(0)
