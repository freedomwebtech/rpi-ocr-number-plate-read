import cv2
import numpy as np
import pandas as pd
import os
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cvzone
from datetime import datetime
from tracker import*
# Initialize YOLO model and PaddleOCR
model = YOLO('yolov10n_float32.tflite')
ocr = PaddleOCR()

# Define the output folder and create it if it doesn't exist
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the region of interest (ROI) for detection

# Load class list from file
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Open video file
cap = cv2.VideoCapture('tc.mov')
processed_numbers = set()
list1 = []

cy1=131
offset=17

# Open file to write car plate data
with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDate\tTime\n")  # Writing column headers

def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
        for line in results[0]:
            text = line[1][0]
            detected_text.append(text)
      
    # Join all detected texts into a single string
    return ''.join(detected_text)
    

count=0
while True:
    ret, frame = cap.read()
    count += 1
    if count % 5 != 0:
        continue

    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame, imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
       
        if 'car' in c:
    
        
           cx=int(x1+x2)//2
           cy=int(y1+y2)//2
           
           if cy1<(cy+offset) and cy1>(cy-offset):
              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
              crop = frame[y1:y2, x1:x2]
              text = perform_ocr(crop)
              print(text)
              if len(text)>5:
                 cvzone.putTextRect(frame, f'{text}', (cx - 50, cy - 30), 1, 1)
                 current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                 with open("car_plate_data.txt", "a") as file:
                      file.write(f"{text}\t{current_datetime}\n")

            # Display detected text on the frame
               

           
            

 
    cv2.line(frame,(2,131),(1018,131),(0,0,255),1)
    cv2.imshow('FRAME', frame)
    if cv2.waitKey(0) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
