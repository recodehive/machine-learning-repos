# QR Code Scanner

This project is a simple QR code scanner that uses OpenCV to capture video from a webcam and detects QR codes in real-time. Once a QR code is detected, the URL or data encoded in the QR code will automatically be opened in your default web browser.

## Features
- Real-time QR code detection using a webcam.
- Automatically opens the decoded URL in the default web browser.
- Exits the application after a QR code is successfully scanned and opened.

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.x
- OpenCV (`cv2`)
- A webcam (integrated or external)

### Python Dependencies
You can install the necessary dependencies using the following commands:

```bash
pip install opencv-python
```

## How to Run the Project
1. Clone or download the project to your local machine.
2. Navigate to the project directory.
3. Ensure you have a working webcam connected.
4. Run the Python script:
```bash
python qr_scan.py
```
The webcam will start, and as soon as a QR code is detected, its URL will be opened in your default web browser.

## Screenshots
- Here is an example of the QR code scanner in action:
    1. **Camera Window Capturing QR Code:**
   ![QR Code Scanner Capturing](https://github.com/ananas304/machine-learning-repos/blob/main/Algorithms%20and%20Deep%20Learning%20Models/QR-Scanner/QR_Scanne-qr%20code%20image%20capture.png)

   In this screenshot, the camera window is open, and it's capturing a QR code in real-time.

## Notes
- The application will exit automatically after a QR code is detected and the link is opened.
- Press the q key to exit the scanner manually at any time.
