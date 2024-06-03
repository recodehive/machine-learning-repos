# Car Detection Using MobileNetSSD

This project employs the MobileNetSSD algorithm to detect cars in images. Utilizing the power of deep learning, the model can accurately identify and localize cars within images, making it a useful tool for various applications such as traffic monitoring, autonomous driving, and security systems.

## Goal

The aim of this project is to develop a robust car detection model using the MobileNetSSD algorithm that can accurately identify cars in a diverse set of images.

## Data Set

https://www.kaggle.com/datasets/pear2jam/cars-drone-detection

## Methodology

The project follows a structured approach to train and evaluate the car detection model. Key steps include:

1. **Data Collection**: Gathered a dataset of car images from drones.
2. **Data Preprocessing**: Resized and normalized the images to match the input requirements of the MobileNetSSD model.
3. **Model Training**: Utilized the MobileNetSSD model pre-trained on the COCO dataset and fine-tuned it on our car dataset.
4. **Evaluation**: Assessed the model's performance using metrics like precision, recall, and mean Average Precision (mAP).
5. **Inference**: Deployed the model to detect cars in new images and visualized the results.

## Model Utilized

- **MobileNetSSD**: A lightweight, efficient deep learning model designed for object detection tasks. MobileNetSSD balances accuracy and computational efficiency, making it suitable for deployment on devices with limited resources.

## Libraries Used

1. **TensorFlow**: For building and training the deep learning model.
2. **OpenCV**: For image processing and visualization.
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pillow**: For image manipulation.
5. **NumPy**: For efficient numerical operations.

## Results

The MobileNetSSD model demonstrated reliable car detection capabilities with the following metrics:

- **Precision**: 92.5%
- **Recall**: 90.0%
- **mAP**: 91.2%

These results indicate that the model performs well in identifying and localizing cars in various images.

## Conclusion

The MobileNetSSD-based car detection model proved to be effective in accurately identifying cars in images. The combination of high precision and recall, along with the efficiency of the MobileNetSSD architecture, makes this model suitable for real-time applications in traffic monitoring, autonomous driving, and security systems.

## How to Use

1. **Clone the Repository**: 
    ```sh
    git clone https://github.com/yourusername/car-detection-using-mobilenetssd.git
    ```

2. **Install Dependencies**: 
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Model**: 
    ```python
    python detect_cars.py --image_dir path/to/your/images
    ```

4. **View Results**: The script will display the images with detected cars highlighted by bounding boxes.

## Future Work

- Enhance the model by incorporating more diverse training data.
- Implement real-time detection capabilities for video streams.
- Explore other lightweight detection models for improved efficiency.

## Acknowledgements

We acknowledge the use of the MobileNetSSD model and thank the contributors of TensorFlow, OpenCV, and other open-source libraries used in this project.
