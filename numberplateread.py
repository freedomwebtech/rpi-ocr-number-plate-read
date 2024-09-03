import cv2
import numpy as np
import pandas as pd
import os
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cvzone

# Initialize YOLO model and PaddleOCR
model = YOLO('best_float32.tflite')
ocr = PaddleOCR()

# Define the output folder and create it if it doesn't exist
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the region of interest (ROI) for detection
area = [(515,123), (512,171), (901, 150), (903,114)]
area1=[(37,300),(48,344),(276,342),(282,308)]
# Load class list from file
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Open video file
cap = cv2.VideoCapture('tc.mov')
count = 0


def perform_ocr(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Perform OCR on the image
    result = ocr.ocr(image, rec=True)  # rec=True enables text recognition

    detected_texts = []
    
    # Process OCR results
    if result:
        for line in result[0]:
            text = line[1][0]
            detected_texts.append(text)
#            print(text)  # Print the detected text
           
    return detected_texts

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
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2

#        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        if result >= 0:
#           cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
#        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
           crop = frame[y1:y2, x1:x2]
           crop=cv2.resize(crop,(140,60)) 
            # Save the cropped image
           image_filename = os.path.join(output_folder, f'frame_{count}_{index}.jpg')
           cv2.imwrite(image_filename, crop)


                # Perform OCR on the saved image
           text=perform_ocr(image_filename)

         
           cvzone.putTextRect(frame, f'{text}'.replace("'"," " ).replace("["," ").replace("]"," "), (x1, y1 -10),1,1)
        result1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
        if result1 >= 0:
#           cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
#           cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
           crop1 = frame[y1:y2, x1:x2]
           crop1=cv2.resize(crop1,(140,60)) 
            # Save the cropped image
           image_filename1 = os.path.join(output_folder, f'frame_{count}_{index}.jpg')
           cv2.imwrite(image_filename1, crop1)


                # Perform OCR on the saved image
           text1=perform_ocr(image_filename1)
           cvzone.putTextRect(frame, f'{text1}'.replace("'"," " ).replace("["," ").replace("]"," "), (x1, y1 -10),1,1)

   
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 1)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 1)

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
