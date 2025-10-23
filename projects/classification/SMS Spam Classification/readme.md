# Email Spam Classification

This repository contains a Streamlit app for classifying email messages as spam or not spam using a machine learning model.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Overview

The Email Spam Classification app uses a trained machine learning model to predict whether an email message is spam or not. The app is built with Streamlit and deployed on Streamlit Cloud.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/navyabijoy/email-spam-classification.git
   cd email-spam-classification
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app:**

   ```sh
   streamlit run main.py
   ```

2. **Interact with the app:**

   Open your browser and go to `http://localhost:8501` to interact with the app. Enter an email message in the text area and click the "Predict" button to see if it is classified as spam or not.

## Model Training

If you want to train the model yourself, follow these steps:

1. **Prepare the dataset:**

   Ensure you have a dataset of email messages labeled as spam or not spam.

2. **Train the model:**

   Use the provided Jupyter notebook or script to train a new model. Save the trained model as `model.pkl` and the vectorizer as `vectorizer.pkl`.

## Deployment

To deploy the app on Streamlit Cloud:

1. **Push your code to GitHub:**

   Ensure your repository is up to date on GitHub.

2. **Deploy on Streamlit Cloud:**

   Follow the [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-cloud) to deploy your app.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.
