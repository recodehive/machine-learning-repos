import cv2
import webbrowser

# Function to open web browser with the decoded QR code link
def open_link_once(url):
    webbrowser.open(url, new=1)

# Open the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

# Initialize the QRCode detector
detector = cv2.QRCodeDetector()

while True:
    success, frame = cam.read()

    # Detect and decode the QR code
    data, bbox, _ = detector.detectAndDecode(frame)

    if data:
        print(f"QR Code data: {data}")

        # Open the URL in a web browser only once
        open_link_once(data)
        break  # Break out of the loop after opening the link

    # Display the camera frame
    cv2.imshow('QR Scanner', frame)

    # Wait for key press and break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
