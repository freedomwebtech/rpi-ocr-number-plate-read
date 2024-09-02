import cv2
from paddleocr import PaddleOCR

def perform_ocr(image_path):
    """
    Perform OCR on the given image and print the detected text.

    Args:
        image_path (str): Path to the image file.

    Returns:
        List of detected texts.
    """
    # Initialize OCR
    ocr = PaddleOCR()

    # Read the image using OpenCV
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
            print(text)  # Print the detected text

    return detected_texts

# Example usage
#mage_path = '/home/pi/Downloads/yolov8-students-counting-lobby-main/output_images/frame_195.jpg'
#texts = perform_ocr(image_path)
