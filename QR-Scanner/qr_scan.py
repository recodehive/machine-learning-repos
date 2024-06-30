import cv2
from pyzbar.pyzbar import decode
import webbrowser

# Function to open web browser with the decoded QR code link
def open_link_once(url):
    webbrowser.open(url, new=1)

# Open the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

while True:
    success, frame = cam.read()

    # Decode QR codes
    for barcode in decode(frame):
        # Extract barcode data
        qr_data = barcode.data.decode('utf-8')
        print(f"QR Code data: {qr_data}")

        # Open the URL in a web browser only once
        open_link_once(qr_data)
        break  # Break out of the for loop after opening the link

    # Display the camera frame
    cv2.imshow('QR Scanner', frame)

    # Check if link has been opened and break out of the main loop
    if 'qr_data' in locals():
        break

    # Wait for key press and break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
