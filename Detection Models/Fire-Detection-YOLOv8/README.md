
# Fire Detection using YOLOv8

## :red_circle: Title
Fire Detection using YOLOv8

## :red_circle: Aim
The aim of this project is to implement a fire detection system using the YOLOv8 (You Only Look Once) object detection model to identify fire in images and videos.

## :red_circle: Brief Explanation
This project leverages the YOLOv8 architecture, which is known for its speed and accuracy in object detection tasks. The model is trained to detect fire and can be applied in various domains, including safety monitoring and environmental protection. 

### Key Features:
- **Real-Time Fire Detection**: Quickly identifies fire in images and videos.
- **Model Implementation**: Easy integration of YOLOv8 for fire detection tasks.
- **Flexible Usage**: Can be adapted for different environments and scenarios.

### Requirements
To run this project, you will need the following dependencies:
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rugved0102/Fire-Detection-using-YOLOv8.git
   cd Fire-Detection-using-YOLOv8/Fire-Detection-YOLOv8
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
You can use the `main.py` file to implement fire detection using YOLOv8. Below is a brief explanation of how to use it:

1. Prepare your input images.
2. Run the following command:
   ```bash
   python main.py --input <path_to_your_image_or_video> --output <path_to_save_output>
   ```
3. The output will display the detected fire instances along with the confidence score.

### Example
```bash
python main.py --input path/to/image.jpg --output path/to/output.jpg
```

## Screenshots 📷
![image](https://github.com/user-attachments/assets/415f5dfe-ee33-42b4-aa8e-2466ad8c6d45)


## Contributions
Feel free to contribute to this project. Please adhere to the following guidelines while making contributions:
- Follow PEP 8 standards for code.
- Use meaningful commit messages.
- Ensure the code works as expected before submitting a pull request.
- Comment your code for better readability.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

