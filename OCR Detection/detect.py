import easyocr
import torch
import cv2

from pathlib import Path


CONFIDENCE_THRESHOLD = 0.3

def detect_text(frame):
    '''
    Detects text from the image.

    Args:
        frame: The frame to detect text from.
    
    Returns:
        image_data: The data of the image generated (bounding_box, text, confidence).
    '''

    reader = easyocr.Reader(['en'], model_storage_directory=str(Path(__file__).resolve().parent / 'Model_Data'), verbose=False)
    image_data = reader.readtext(frame, paragraph=False)

    return image_data


def process_data(frame, image_data, create=True):
    '''
    Create bounding boxes around the detected text, and returns the detected text in lowercase.

    Args:
        frame: The frame to draw the bounding boxes on.
        image_data: The data of the image generated (bounding_box, text, confidence).
        create: If True, the bounding boxes will be drawn on the frame.
    
    Returns:
        detected_text: The detected text in lowercase.
    '''

    detected_text = ""

    for (bbox, text, prob) in image_data:
        if prob > CONFIDENCE_THRESHOLD:
            if create:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                cv2.putText(frame, f'{text} - {int(prob * 100)}%', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected_text += text.strip() + " "
    
    return detected_text.strip().lower()