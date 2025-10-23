# 🍲 Food Allergy Prediction

Welcome to the **Food Allergy Prediction** project! This project aims to build a machine learning model that predicts food allergies based on various features such as ingredients, demographic details, and medical history. The model is designed to assist healthcare professionals and individuals in managing food allergies effectively.

## 📋 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Introduction

Food allergies are a major concern for millions of people worldwide. Accurate prediction and management of these allergies can significantly improve the quality of life. This project leverages **machine learning** techniques to identify and predict food allergies, providing insights based on historical data.

## ✨ Features

| Feature               | Description                                            |
|-----------------------|--------------------------------------------------------|
| 📊 **Data Analysis**  | Exploratory Data Analysis (EDA) to identify patterns   |
| 🤖 **Machine Learning**| ML model to predict food allergies based on inputs    |
| 📈 **Visualization**  | Graphical representation of data for better insights   |
| 🏥 **Health Focus**   | Tailored predictions for various demographics          |

## 📚 Dataset

The dataset used in this project includes:
- **Demographic Details:** Age, gender, etc.
- **Food Details:** Ingredients and nutritional information.
- **Medical History:** Existing conditions, past allergic reactions.

> **Note:** The dataset is preprocessed and cleaned for accurate predictions. Make sure to review the data structure in the notebook for detailed insights.

## 🛠️ Installation

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/alo7lika/food-allergy-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd food-allergy-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## 🏗️ Model Architecture

The model leverages a combination of **Random Forest** and **XGBoost** algorithms for high accuracy. The architecture includes:

- **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables.
- **Model Training**: Using cross-validation to find the best hyperparameters.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1 score.

| Step                | Description                                 |
|---------------------|---------------------------------------------|
| 1️⃣ Data Cleaning    | Removing inconsistencies in the dataset     |
| 2️⃣ Feature Engineering | Creating meaningful features               |
| 3️⃣ Model Training   | Training using Random Forest and XGBoost    |
| 4️⃣ Evaluation       | Measuring model performance                 |

## 🚀 Usage

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the notebook file `Food Allergy Prediction.ipynb`.
3. Run the cells step by step to train the model and see the results.

> **Tip:** You can customize the dataset and re-train the model for better performance based on specific use cases.

## 📊 Results

| Metric       | Value   |
|--------------|---------|
| **Accuracy** | 95.2%   |
| **Precision**| 92.8%   |
| **Recall**   | 93.5%   |
| **F1 Score** | 93.1%   |

The model achieved **high accuracy** and provides robust predictions across different demographic groups and food types.

## 🤝 Contributing

We welcome contributions to enhance the project! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-branch
   ```
5.Open a pull request.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

