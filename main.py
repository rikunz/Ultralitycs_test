import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from tracker import*
from pathlib import Path
import cvzone

def point_in_polygon(x, y, polygon):
    """
    Check if point (x,y) is inside the polygon using ray casting algorithm
    """
    inside = False
    n = len(polygon)
    j = n - 1
    
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and 
            (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / 
             (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i
    
    return inside

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('Tangga.mp4')
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1020, 500))
class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


count=0

tracker=Tracker()

area1=[(596,314),(566,318),(804,385),(812,364)]
area2=[(268,143),(243,143),(236,243),(300,270),(350,266),(266,237)]
area3 = [(55,243),(3,246),(5,491),(1000,490),(902,436),(270,356)]
exclude_area =[(330,4),(330,180),(830,209),(846,4)]
exclude_area2 = [(1,156),(1,230),(40,235),(44,166)] 


people_up = {}
people_down = {}
people_floor = {}
counter_up = []
counter_down = []
counter_floor = []
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]         
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c and not point_in_polygon(x2, y2, exclude_area) and not point_in_polygon(x2,y2, exclude_area2):
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
        cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
        cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
        #UP
        results = cv2.pointPolygonTest(np.array(area1,np.int32), (x4,y4), False)
        if results >=0:
            people_up[id] = (x4,y4)
        if id in people_up:
            going_up_from_down = cv2.pointPolygonTest(np.array(area2,np.int32), (x4,y4), False)
            going_floor_from_down = cv2.pointPolygonTest(np.array(area3,np.int32), (x4,y4), False)
            if going_up_from_down >= 0:
                if counter_up.count(id) == 0:
                    counter_up.append(id)
            if going_floor_from_down >=0:
                if counter_floor.count(id) == 0:
                    counter_floor.append(id)
        # Down
        results2 = cv2.pointPolygonTest(np.array(area2,np.int32), (x4,y4), False)
        if results2 >=0:
            people_down[id] = (x4,y4)
        if id in people_down:
            going_down_from_up = cv2.pointPolygonTest(np.array(area1,np.int32), (x4,y4), False)
            going_floor_from_up = cv2.pointPolygonTest(np.array(area3,np.int32), (x4,y4), False)
            if going_down_from_up >= 0:
                if counter_down.count(id) == 0:
                    counter_down.append(id)
            if going_floor_from_up >= 0:
                if counter_floor.count(id) == 0:
                    counter_floor.append(id)
        #Floor
        results3 = cv2.pointPolygonTest(np.array(area3,np.int32), (x4,y4), False)
        if results3 >=0:
            people_floor[id] = (x4,y4)
        if id in people_floor:
            going_down_from_floor = cv2.pointPolygonTest(np.array(area1,np.int32), (x4,y4), False)
            going_up_from_floor = cv2.pointPolygonTest(np.array(area2,np.int32), (x4,y4), False)
            if going_down_from_floor >= 0:
                if counter_down.count(id) == 0:
                    counter_down.append(id)
            if going_up_from_floor >= 0:
                if counter_up.count(id) == 0:
                    counter_up.append(id)
                    
        

            
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),1)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(255,0,0),1)
    cvzone.putTextRect(frame,f'Up:{len(counter_up)}',(50,60),1,1,font=cv2.FONT_HERSHEY_COMPLEX_SMALL,colorR=(0,0,255))
    cvzone.putTextRect(frame,f'Floor:{len(counter_floor)}',(50,90),1,1,font=cv2.FONT_HERSHEY_COMPLEX_SMALL,colorR=(0,0,255))
    cvzone.putTextRect(frame,f'Down:{len(counter_down)}',(50,120),1,1,font=cv2.FONT_HERSHEY_COMPLEX_SMALL,colorR=(0,0,255))
    cv2.putText(frame,'People Counter In Floor', (340,55),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
    cvzone.putTextRect(frame, "Thoriq Firdaus Arifin", (725,460),1,1,font=cv2.FONT_HERSHEY_COMPLEX_SMALL,colorT=(0,0,0),colorR=(255,255,255))
   

 
    cv2.imshow("RGB", frame)
    video_writer.write(frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
video_writer.release()
cv2.destroyAllWindows()

#()