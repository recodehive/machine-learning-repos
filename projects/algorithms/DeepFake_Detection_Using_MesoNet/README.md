

---
# Deepfake Detection using MesoNet with PyTorch

## Overview

This project implements deepfake detection using MesoNet, a convolutional neural network (CNN) designed specifically for detecting deepfake images. The model is trained on a dataset containing both real and deepfake images and is deployed for real-world deepfake image detection.

## Features
- Utilizes MesoNet architecture for deepfake image detection.
- Preprocessing techniques for image dataset preparation.
- Training and evaluation procedures for model development.
- Deployment for real-world deepfake image detection.
- Flask application for interactive deepfake image detection via a web interface.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Flask
- NumPy
- Matplotlib
- Jupyter Notebook (optional, for training visualization)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/saikrishna823/DeepFake_Detection_Using_MesoNet.git
   ```

2. Create and activate a Python virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # For Unix/Linux
   venv\Scripts\activate      # For Windows
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation:**

   - Prepare a dataset containing real and deepfake images.
   - Ensure proper labeling and preprocessing of the image dataset.

2. **Model Training:**

   - Train the MesoNet model using the provided training script.
   - Adjust hyperparameters and training configurations as needed.

3. **Model Evaluation:**

   - Evaluate the trained model using the provided evaluation script.
   - Analyze performance metrics such as accuracy, precision, recall, etc.

4. **Deployment:**

   - Deploy the trained model for real-world deepfake image detection.
   - Integrate the model into an application or platform for automated detection.

5. **Flask Application:**

   - Navigate to the `app` directory.
   - Run the Flask application:

     ```
     flask run
     ```

   - Access the deepfake image detection web interface in your browser at `http://localhost:5000`.

## Contributing

Contributions are welcome! Please follow the standard GitHub workflow:

1. Fork the repository.
2. Create a new branch 
3. Make your changes.
4. Commit your changes 
5. Push to the branch 
6. Create a new Pull Request.


## Acknowledgements

- Dataset used for training: https://zenodo.org/record/5528418#.YpdlS2hBzDd.

## Contact

For inquiries or support, please contact:saikrishnareddymule200@gmail.com.

