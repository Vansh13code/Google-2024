import cv2
from ultralytics import YOLO
model=YOLO("webcam/best (5).pt")
results=model("samosa.mp4",show=True)
cv2.waitKey(0)
