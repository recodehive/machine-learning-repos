# Video Tampering and Forensics Analysis Tool

## Abstract

The **Video Tampering and Forensic Analysis Tool** is designed to ensure the authenticity and integrity of digital video content. In fields like law enforcement and journalism, reliable video evidence is crucial. This tool uses advanced techniques such as metadata examination, frame-by-frame analysis, deepfake detection, and watermark authentication to detect video tampering and manipulations. It generates detailed forensic reports to aid investigative professionals in ensuring the credibility of video evidence.

## Features

- **User-Friendly Interface**: Interactive web-based application using Streamlit.
- **Video Input**: Supports video file formats such as MP4, AVI, and MOV (file size limit: 200MB).
- **Metadata Extraction**: Provides key details such as resolution, codec, frame count, duration, and geolocation data (if available).
- **Hashing (MD5)**: Verifies the integrity of the video by generating a unique hash for comparison.
- **Frame Analysis**: Detects tampering through frame-by-frame inspection for splicing, insertion, and pixel pattern anomalies.
- **Comprehensive Report Generation**: Outputs a JSON report with all findings, including extracted metadata and hash values.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: OpenCV, FFmpeg, hashlib, os, json
- **Web Framework**: Streamlit
- **Hashing Algorithm**: MD5

## Setup Instructions

#### First Fork the repository and then follow the steps given below!

### 1. Clone the Repository

```sh
git clone https://github.com/<your-username>/machine-learning-repos.git
cd Detection Models/Video-tampering-detection
```
### 2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

### 3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### 4. Run the application:
  ```sh
    streamlit run main.py
  ```
### 5. Usage Instructions

  ##### 1. Open your web browser and go to `http://localhost:8501` to access the app.
  
  ##### 2. Upload a video file (MP4, AVI, MOV) for analysis.

  ##### 3. View the results on the web interface, including:
  - Extracted metadata
  - MD5 hash for integrity verification
  - Frame-by-frame analysis

  ##### 4. Download the comprehensive JSON report summarizing the findings.

### 6. Stop the Application:
  To stop the Streamlit app, press `CTRL + C` in the terminal.


## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or create a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
