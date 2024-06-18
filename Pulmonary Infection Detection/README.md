# pulmonary-infection-detection
An application that receives the employee’s respiratory audio as input and the trained model will analyze the audio and output what kind of infection is being affected to the employee.

## Problem

 
In food processing industries, ensuring the health and safety of workers is of utmost importance. Respiratory problems pose a significant risk to workers in food processing industries, where exposure to airborne particles and contaminants is common. Early detection of respiratory issues is crucial for preventing long-term health complications and ensuring the well-being of employees. 

## Solution 

We have created an AI model that receives the employee’s respiratory audio as input and the trained model will analyze the audio and output what kind of infection is being affected to the employee. The Pulmonary Infection Detection is a deep learning-based system designed to analyze audio recordings of respiratory sounds and classify them into different respiratory diseases. This project encompasses various stages from data collection to model deployment, providing a comprehensive solution for diagnosing respiratory ailments.

## Features

- Early Detection: By analyzing cough sounds, respiratory issues can be detected at an early stage, allowing for prompt medical attention and intervention. This can prevent the progression of respiratory diseases and minimize their impact on workers' health.
- Real-time Monitoring: AI-based cough analysis systems can provide real-time insights into workers' respiratory health status, enabling proactive management and mitigation of occupational health risks. Continuous monitoring allows for prompt identification of emerging issues and implementation of preventive measures.
- Improved Productivity: By maintaining a healthier workforce, the implementation of respiratory disease detection can help minimize absenteeism due to respiratory illnesses. Healthy workers are more productive and contribute to the efficient operation of food processing facilities.Enhanced 
- Food Safety: Respiratory health monitoring can contribute to maintaining high food safety standards by reducing the risk of contamination from sick workers. Ensuring the health of workers helps prevent the spread of pathogens and allergens in food processing environments.

## Workflow

1. **Data Curation**: Gathering and organizing audio data along with patient diagnosis information.
2. **Data Processing**: Preprocessing audio data to extract relevant features for classification.
3. **Data Splitting**: Partitioning the dataset into training, validation, and testing sets for model evaluation.
4. **CNN Training**: Training a Convolutional Neural Network (CNN) to learn features from the audio data.
5. **Model Evaluation**: Assessing the performance of the trained model using validation and test datasets.
6. **Building a Web App**: Developing a user-friendly web application for real-world usage of the classifier.

## Key Dependencies

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Librosa
- Seaborn
- Resampy

## Features

- **Data Augmentation**: Various techniques like adding noise, shifting, stretching, and pitch shifting applied to augment the dataset.
- **Model Architecture**: Utilizes a combination of Convolutional Neural Networks (CNN) and Gated Recurrent Units (GRU) for effective feature extraction and classification.
- **Model Training and Evaluation**: Trains the model on augmented data and evaluates its performance using metrics like accuracy, precision, recall, F1 score, and confusion matrix.
- **Web App Deployment**: Provides a user-friendly interface for users to upload audio recordings and obtain disease diagnosis predictions.

## Usage

1. **Data Collection**: Gather respiratory sound recordings along with patient diagnosis information.
2. **Data Preprocessing**: Apply necessary preprocessing steps such as feature extraction and data augmentation.
3. **Model Training**: Train the respiratory disease classifier using the provided dataset.
4. **Model Evaluation**: Assess the performance of the trained model using validation and test datasets.
5. **Deployment**: Deploy the trained model into a web application for real-world usage.






