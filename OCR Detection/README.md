# OCR_Detection

The OCR Detection is introduced in order to help the victims in the situation where they can not use other means to communicate or seek help, other than
written communication shown to the camera. Also, it helps in detecting potential self-harm when the victim is in the process of writing the death note, and the 
camera catches a glimpse of it and can use existing models to determine the scale of the threat that uses text as their primary input.

# Requirements
- `pip install -r requirements.txt`.

# Usage
**It is to be kept in mind that a window will only be created if there are text detected by the model. For visualizing another image, that window has to be closed in order for the another window to appear.**

- `demo.py` to start the web camera for obtaining frames.
- Ctrl + C to exit from the script.
- If a text is detected, a new window opens with the text detected, annotations and the confidence. The detected text is also printed for convenience.

# Working
`easyocr` package is used to provide image to text detection. `Model_Data` contains the downloaded model to reduce the online dependancy.

- `detect.py` contains the functions that can be imported by other scripts to be executed to perform image to text detection.
- `demo.py` contains a demo code which showcases the functionality.

OpenCV without GUI (`opencv-python-headless`) is used to optimize the script for detection. It is useful in optimizing the detection speed by removing useless processes used for GUI.
Additionally, for web integration, GUI is not needed but the other functionalities remains the same.

`demo.py` also contains an optimization which prevents the execution of the model detection if the frame difference is less, i.e., the frames hasn't changed much. MSE (Mean Squared Error) is used to calculate the difference between the two frame. The model only gets executed, if the error is greater than `20`. This can be modified by changing the value of `ERR_DIFF`.

Multi-processing can be used to get seamless detections without delay.

# Demo
- The image is purely for the demonstration purposes:

![Figure_1](https://github.com/SAM-DEV007/ThereForYou/assets/60264918/86f3abd0-5fe6-4c23-ad6f-22300058eca2)

- The more the resolution of the camera, the better the results.
