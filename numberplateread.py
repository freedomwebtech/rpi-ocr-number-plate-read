import cv2
import pandas as pd
import os
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cvzone

# Initialize YOLO model and PaddleOCR
model = YOLO('best_float32.tflite')
ocr = PaddleOCR()


# Load class list from file
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Open video file
cap = cv2.VideoCapture('nr.mp4')

cy1 = 415
offset = 20



def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
#        print(results)
        for result in results[0]:
            print(result)
            text = result[1][0]
            detected_text.append(text)
      
    # Join all detected texts into a single string
    return ''.join(detected_text)

count = 0

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
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if cy1<(cy+offset) and cy1>(cy-offset):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crop = frame[y1:y2, x1:x2]
            crop=cv2.resize(crop,(110,30))
            text = perform_ocr(crop)
#            print(text)
            cvzone.putTextRect(frame, f'{text}', (cx - 50, cy - 30), 1, 1)

           

    cv2.line(frame,(0,415),(1018,415),(255,255,255),1)
    cv2.imshow('FRAME', frame)
    if cv2.waitKey(0) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
