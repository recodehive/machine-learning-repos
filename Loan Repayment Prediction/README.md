# Loan Repayment Prediction Project

## Overview

Welcome to the Loan Prediction Project! This project aims to build a predictive model to identify the likelihood of a loan application being approved. Leveraging machine learning techniques, this project helps financial institutions streamline their loan approval processes and minimize risks.

## Table of Contents

- [Project Description](#project-description)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Description

The Loan Prediction Project utilizes various machine learning algorithms to predict loan approvals based on applicant information. The primary goal is to create a model that accurately predicts whether a loan should be approved, thus aiding financial institutions in decision-making.

## Data Description

The dataset used in this project contains information about loan applicants, including:

- **Applicant Information**: Gender, Marital Status, Education, Number of Dependents, etc.
- **Financial Information**: Applicant Income, Co-applicant Income, Loan Amount, Loan Amount Term, Credit History, etc.
- **Loan Information**: Loan ID, Loan Status, Property Area, etc.

## Installation

To run this project, you'll need to have Python installed. Follow the steps below to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/aviralgarg05/Loan-Prediciton-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Loan-Prediciton-Project
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the project, follow these steps:

1. Preprocess the data by running the preprocessing script:
   ```bash
   python preprocess.py
   ```
2. Train the model using the training script:
   ```bash
   python train_model.py
   ```
3. Evaluate the model using the evaluation script:
   ```bash
   python evaluate_model.py
   ```

## Modeling

This project explores various machine learning models, including:

- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine**

Each model is evaluated based on its accuracy, precision, recall, and F1 score. The best-performing model is selected for predicting loan approvals.

## Results

1. The Loan Status is heavily dependent on the Credit History for Predictions.
2. The Logistic Regression algorithm gives us the maximum Accuracy (79% approx) compared to the other Machine Learning Classification Algorithms.

The final model demonstrates strong predictive power.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Create a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or suggestions, please feel free to contact:

- **Name**: Aviral Garg
- **Email**: [gargaviral99@gmail.com](mailto:gargaviral99@gmail.com)
- **GitHub**: [aviralgarg05](https://github.com/aviralgarg05)
