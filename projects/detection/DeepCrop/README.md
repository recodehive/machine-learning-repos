# DeepCrop

## Description
A machine learning project for crop disease detection using deep learning. The system can identify healthy crops and detect diseases like Early Blight and Late Blight in potato plants.

## Project Structure
```
DeepCrop/
├── data/              # Dataset files and test images
├── notebooks/         # Jupyter notebooks for experimentation
├── src/              # Source code (Flask API)
├── models/           # Saved trained models
├── frontend/         # React frontend application
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## Dataset
The project uses a crop disease dataset with multiple classes including:
- Healthy
- Early Blight
- Late Blight
- Other disease categories

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Backend API
Start the Flask server:
```bash
cd src
python app.py
```

The API will run on `http://localhost:5000`

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### API Endpoints
- `POST /predict` - Upload an image for disease prediction
- `GET /` - Health check endpoint

## Model Details
- **Algorithm used:** Convolutional Neural Network (CNN) with Keras/TensorFlow
- **Input size:** 128x128 pixels
- **Classes:** 3 main classes (Healthy, Early Blight, Late Blight)
- **Framework:** TensorFlow/Keras

## Results
The model provides:
- Disease classification
- Confidence percentage
- Support for three main crop health categories

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License
