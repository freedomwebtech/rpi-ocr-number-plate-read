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
area = [(23, 83), (39, 457), (714, 467), (708, 83)]

# Load class list from file
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Open video file
cap = cv2.VideoCapture('nread.mp4')
count = 0

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

def perform_ocr(image_array):
    """
    Perform OCR on the given image array.
    
    Parameters:
    image_array (numpy.ndarray): The image on which OCR needs to be performed.
    
    Returns:
    list: A list of detected text strings.
    """
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    result = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition

    detected_texts = []

    # Process OCR results
    if result and result[0] is not None:
        for line in result[0]:
            text = line[1][0]
            detected_texts.append(text)
    else:
        print(f"No OCR results found for the provided image.")

    return detected_texts

while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
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

        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        if result >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]

            # Perform OCR directly on the cropped image array
            text = perform_ocr(crop)

            # Display detected text on the frame
            cvzone.putTextRect(frame, f'{text}'.replace("'", " ").replace("[", " ").replace("]", " "), (x1, y1 - 10), 1, 1)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)

    cv2.imshow('RGB', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
