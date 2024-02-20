from ultralytics import YOLO
import cv2
import cvzone
import math
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 960)
model=YOLO("best (5).pt")
classNames=['Chicken Chukka - 750cal - Unhealthy', 'Chicken Curry-248Cal-Healthy', 'Fish Head Curry - 1183cal - Healthy', 'Parota-480Cal-Unhealthy', 'Prawn Curry - 708cal - Healthy', 'Samosa-308Cal-Unhealthy', 'South Indian Crab Curry-1565Cal-Healthy', 'South Indian Fish Curry-766 calories-Healthy', 'Urad Vadai-52 calories-Healthy', 'idli-60calor9ies-healthy']
while True:
    success,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2 =int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_names=classNames[cls]
            label=f'{class_names}{conf}'
            t_size=cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]
            c2=x1+t_size[0],y1-t_size[1]-3
            cv2.rectangle(img,(x1,y1),c2,[255,0,255],-1,cv2.LINE_AA)
            cv2.putText(img,label,(x1,y1-2),0,1,[255,255,255],thickness=1,lineType=cv2.LINE_AA)


    cv2.imshow("Image",img)
    cv2.waitKey(1)