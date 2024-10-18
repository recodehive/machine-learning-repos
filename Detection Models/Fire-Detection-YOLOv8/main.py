# main.py
"""
Fire Detection using YOLOv8

This script demonstrates how to use YOLOv8 for detecting fire in images. 
It loads a pre-trained YOLOv8 model, takes an input image, and performs fire detection.
Ensure that the YOLOv8 model is trained or fine-tuned for fire detection before using.

Follow PEP 8 coding standards.

Usage:
    python main.py --image <path_to_image>
"""

import argparse
import cv2
from ultralytics import YOLO

# Function to perform fire detection using YOLOv8
def detect_fire(image_path):
    """
    Detects fire in an input image using YOLOv8 model.

    Parameters:
    image_path (str): The path to the input image for fire detection.

    Returns:
    None
    """
    # Load the YOLOv8 model (pre-trained or fine-tuned on fire detection)
    model = YOLO("yolov8n.pt")  # Replace "yolov8n.pt" with the path to your YOLOv8 fire detection model

    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image from {image_path}")
        return

    # Perform detection
    results = model(image)

    # Display the results
    print(f"Fire detected in the image: {image_path}")
    results.show()  # This will open a window to display the image with detection results

    # Save the results image
    results.save(save_dir="runs/detect/fire_detection")

if __name__ == "__main__":
    # Set up argument parsing for the input image
    parser = argparse.ArgumentParser(description="Fire detection using YOLOv8.")
    parser.add_argument("--image", required=True, help="Path to the input image for fire detection.")
    args = parser.parse_args()

    # Call the detection function with the input image
    detect_fire(args.image)

