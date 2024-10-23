# Frame Extraction and Entropy Calculation

This project processes video files by extracting frames, saving them as images, converting them to grayscale, and calculating entropy for frame selection based on histogram differences. Selected frames are plotted based on their entropy values.

## Features
- Extracts frames from a video file.
- Converts frames to grayscale.
- Detects significant changes between consecutive frames using histogram comparison.
- Calculates and visualizes entropy for selected frames.

## Dependencies

- `opencv-python` (`cv2`)
- `numpy`
- `matplotlib`

You can install the required packages using the following command:

```bash
pip install opencv-python numpy matplotlib
```

## How It Works
**1) Frame Extraction:** The video is processed frame by frame using OpenCV. Each frame is saved in two folders:

- `./vout1/` for the original frames.
- `./grayvout1/` for the grayscale frames.

**2) Histogram Difference Calculation:** The grayscale frames are analyzed using histogram comparison to measure the difference between consecutive frames.

**3) Entropy Calculation:** Entropy is calculated for selected frames based on their pixel distribution. The entropy measures the uncertainty or randomness in the frame.

**4) Frame Selection:** The script sorts the frames based on the histogram differences and selects the top `100` frames with the largest changes.

**5) Entropy Plot:** Finally, the entropy of the selected frames is plotted using `matplotlib`.


## Key Functions
- `calEntropy(image):`
  - This function calculates the entropy of an image by analyzing the pixel distribution.
- `calculate_frame_changes(files):`
  - This function calculates the histogram difference between consecutive frames.
- `select_frames(changes, num_frames=100):`
  - This function selects the frames with the most significant changes based on the histogram differences.

## Output
The script will generate two outputs:

- **Frames:** Saved as PNG files in the specified directories.
- **Entropy Plot:** A graph showing the entropy values of the selected frames.

## Directory Structure
```
├── Sample.mkv            # Input video file
├── frame_entropy_calculator.py   # Main Python script
├── vout1/                # Directory for original frames
├── grayvout1/            # Directory for grayscale frames
└── README.md             # This file
```
