import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR()

# Read image using opencv
image = cv2.imread('/home/pi/Downloads/rpinumberplate-main/cropped_images/34.jpg')

# Perform OCR on the image
result = ocr.ocr(image, rec=True)  # Change rec to True to get text results
#for row in result[0]:
#    print(row)
    
    
for line in result[0]:
    print(line[1][0])