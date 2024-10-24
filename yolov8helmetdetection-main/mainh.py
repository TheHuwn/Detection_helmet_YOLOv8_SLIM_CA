import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import torch
import torch.nn as nn
from slim_neck import SlimNeck  # Import lớp SlimNeck đã tạo

# Mô hình YOLOv8 với SlimNeck
class YOLOv8WithSlimNeck(nn.Module):
    def __init__(self):
        super(YOLOv8WithSlimNeck, self).__init__()
        self.model = YOLO('best2.pt')  # YOLOv8 backbone
        self.neck = SlimNeck(in_channels=256)  # SlimNeck tích hợp

    def forward(self, x):
        results = self.model.predict(source=x)  # Sử dụng hàm predict của YOLOv8
        return results

# Khởi tạo mô hình YOLOv8 với SlimNeck
model = YOLOv8WithSlimNeck()

# Hàm lấy tọa độ của chuột
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)
  
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('he2.mp4')

# Đường dẫn lưu video trong thư mục D:
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('D:/yolov8helmetdetection-main/yolov8helmetdetection-main/output_video.mp4', fourcc, 20.0, (1020, 500))

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (1020, 500))
    
    # Sử dụng mô hình để dự đoán
    results = model(frame)  # Gọi hàm forward() của YOLOv8WithSlimNeck
    a = results[0].boxes.data.cpu().numpy()  # Lấy kết quả và chuyển sang numpy
    px = pd.DataFrame(a).astype("float")
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        # Vẽ khung chữ nhật xung quanh đối tượng phát hiện được
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    
    # Hiển thị frame và ghi vào file video
    cv2.imshow("RGB", frame)
    out.write(frame)  # Ghi frame vào video
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng tài nguyên
cap.release()
out.release()  # Đóng đối tượng ghi video
cv2.destroyAllWindows()
