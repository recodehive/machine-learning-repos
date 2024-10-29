# Face Recognition Using 3 Models

## Overview

This project implements a face recognition system using three different models: 
1. **Model A** (e.g., Eigenfaces)
2. **Model B** (e.g., Fisherfaces)
3. **Model C** (e.g., Convolutional Neural Networks)

The goal is to compare the performance of these models in terms of accuracy, speed, and robustness in recognizing faces from images.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Face detection and recognition
- Comparison of different algorithms
- User-friendly interface
- Visualization of results

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-recognition.git
   cd face-recognition
   ```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Prepare Your Dataset

Prepare your dataset of images. Ensure they are organized in folders by label (e.g., `dataset/person1`, `dataset/person2`).

### 2. Run the Face Recognition Script

Open your terminal and execute the following command:

```bash
python main.py --model [model_name] --input [path_to_image]
```
## Models

### Model A: Eigenfaces

- Based on Principal Component Analysis (PCA)
- Suitable for small datasets
### Visualizations  
![image](https://github.com/user-attachments/assets/9341ea4d-ab6c-4dc6-8690-c2865ed9cdfb)


### Model B: Fisherfaces

- Uses Linear Discriminant Analysis (LDA)
- More robust to variations in lighting and expression
### Visualizations 
![image](https://github.com/user-attachments/assets/585df1b3-2480-4d8c-851a-db52cebb0393)


### Model C: Convolutional Neural Networks (CNNs)

- Utilizes deep learning techniques
- Provides high accuracy but requires more computational power
### Visualizations 
![image](https://github.com/user-attachments/assets/d09f4008-37b8-4162-8928-40250a200ff7)


## Results

| Model   | Accuracy | Speed    | Notes                        |
|---------|----------|----------|------------------------------|
| Model A | 85%      | Fast     | Good for small datasets      |
| Model B | 90%      | Moderate | Better for varied conditions  |
| Model C | 95%      | Slow     | Requires more resources      |

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. Make sure to follow the contribution guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for the computer vision library
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework
