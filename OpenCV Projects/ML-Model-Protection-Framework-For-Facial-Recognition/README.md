## Project Overview

This project addresses the security challenges of deploying ML models for face authentication in a browser context. Our goal is to safeguard the ML model from reverse engineering and tampering while ensuring that security measures do not significantly impact the model’s size or degrade the user experience.

## Description

The application provides a proof of concept for face authentication using TensorFlow.js and FaceMesh in a browser environment. It integrates a Trusted Execution Environment (TEE) for secure handling of the ML model, employing AES-256 encryption for data protection and obfuscation techniques to optimize model size and security.

## How It Works

1. **Face Detection**:
   - The frontend uses TensorFlow.js and the FaceMesh model to detect faces via the webcam.
   - An encrypted message containing the face data is sent to the web server.

2. **Data Handling**:
   - The web server forwards the encrypted data to the TEE server for decryption and authentication.

3. **Model Security**:
   - The TEE server decrypts the data using AES-256 encryption.
   - The model is obfuscated and snapshotting is used to minimize its size and improve load times.

4. **Authentication**:
   - Based on the decrypted data, the TEE server checks if the face authentication is successful.
   - The result is sent back to the web server and then displayed to the client.

## Technologies, Frameworks, and Tools

- **Frontend**:
  - HTML
  - JavaScript
  - TensorFlow.js
  - FaceMesh

- **Backend**:
  - Flask
  - MongoDB (for potential data storage)

- **Security**:
  - AES-256 Encryption
  - Trusted Execution Environment (TEE)
  - Obfuscation
  - Snapshotting

## Unique Selling Points

- **Secure Model Handling**: Utilizes Trusted Execution Environment (TEE) to protect against tampering and reverse engineering.
- **Efficient Encryption**: Employs AES-256 encryption for secure data transmission.
- **Optimized Model Size**: Integrates obfuscation and snapshotting techniques to minimize model size impact.
- **Real-Time Authentication**: Leverages TensorFlow.js and FaceMesh for accurate face detection directly in the browser.
- **Seamless Integration**: Maintains user experience with minimal performance overhead while ensuring robust security.

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. **Start the TEE server**:
    ```bash
    python tee_server.py
    ```

2. **Start the Web server**:
    ```bash
    python web_server.py
    ```

3. **Access the application**:
    Open a web browser and navigate to `http://localhost:5000`. You will see the registration and login pages.

4. **Test the application**:
    - **Register**: Enter a username and click "Register" to save the face image.
    - **Login**: Enter the username and click "Login" to authenticate with the captured face image. Successful login will redirect to the home page.

## Troubleshooting

- **Video not appearing**: Ensure that the webcam is properly connected and permissions are granted for the browser to access it.
- **Issues with image capture**: Check the browser console for errors related to the webcam or canvas.

## Acknowledgments

- Thanks to TensorFlow.js and FaceMesh for their face detection capabilities.
