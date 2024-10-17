# Parking Space Detection

This project allows users to detect available parking spaces in a video feed. The program utilizes OpenCV to process video frames and identifies parking spaces using a pre-defined set of coordinates, which can be manually selected using a mouse-click event. The positions of the parking spaces are saved in a file (`CarParkPos`), and the program then monitors whether each parking space is occupied or vacant.

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Project Description](#project-description)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Usage](#usage)
- [How to Load Your Own Video](#how-to-load-your-own-video)
- [Screenshots](#screenshots)

## Features
- Manually select parking spaces by clicking on the video frame.
- Save the selected parking space coordinates using pickle.
- Monitor the parking spaces in real-time and display the number of vacant spaces.
- Works with any video footage of parking lots.

## Demo
- 🖼️ Demo Video of Parking Detection
  
  [empty space detection.webm](https://github.com/user-attachments/assets/ea920495-edf4-42a8-a212-dbee36e6c76f)


## Project Description

This project uses computer vision techniques with OpenCV to detect available parking spaces in a video feed. The system allows users to manually define the parking spaces in a given video by selecting their positions using mouse clicks. Once the spaces are defined, the program monitors each space in real-time to check if it is occupied or vacant based on pixel intensity changes. The workflow involves processing each frame of the video, comparing the state of the selected parking spaces, and updating their occupancy status accordingly.

The technology stack includes:
- **OpenCV** for video processing and real-time space monitoring.
- **Pickle** for saving and loading the parking space positions.
- **CvZone** for assisting with various computer vision operations.



## File Descriptions

- **`CarParkPos.py`**: This script allows users to manually select parking spaces in the video by clicking on the video frame. The positions of the parking spaces are saved to a file (`CarParkPos`) using the `pickle` module, which stores the coordinates for future use. This file acts as a setup step for defining the parking spaces.

- **`main.py`**: This script monitors the parking spaces in real-time. It reads the positions saved from `CarParkPos.py` and processes each video frame to check whether each defined parking space is occupied or vacant. It then displays the real-time status of each parking space on the video and tracks the number of vacant spaces.


## Requirements
`requirements.txt` contains all the required Python libraries.
```txt
opencv-python
cvzone 
```

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run `CarParkPos.py`
This script allows you to manually select parking spaces by clicking on the video frame.
- **Left Click** on the video to mark the top-left corner of a parking space.
- **Right Click** to remove a previously selected parking space.

The selected positions are saved in a binary file `CarParkPos` using the `pickle` module.

### Step 2: Run `main.py`
This script will use the saved parking positions from the `CarParkPos` file and start monitoring the parking spaces in the video feed. It will display the number of vacant spaces in real-time.

```bash
python main.py
```

## How to Load Your Own Video
To use this project with your own parking lot video:
1. Rename your video file to `carPark.mp4` or modify the code in both `CarParkPos.py` and `main.py` to point to your video filename.
   ```python
   cap = cv2.VideoCapture('your-video-file.mp4')
   ```
2. Ensure the video shows a clear view of the parking lot so you can manually mark parking spaces.

3. Run the `CarParkPos.py` script to mark parking spaces in your video, and then run `main.py` to start monitoring those spaces.

## Screenshots
- **Selecting Parking Spaces**:
Below are screenshots from `CarParkPos.py` to demonstrate how parking spaces can be selected:
![Parking Space Selection](https://github.com/ananas304/machine-learning-repos/blob/main/OpenCV%20Projects/ParkingSpaceDetector/carParkPos.png)

- **Detecting Available Spaces**:
Detecting available spaces in real-time by analyzing pixel changes in the predefined parking regions to determine occupancy status.
![Detecting available spaces](https://github.com/ananas304/machine-learning-repos/blob/main/OpenCV%20Projects/ParkingSpaceDetector/empty%20space%20detection.png)
