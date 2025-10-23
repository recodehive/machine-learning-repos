# Explicit Content Classification Model

This project aims to classify whether a music track is explicit or not based on its audio features using a RandomForestClassifier.

## Overview

The model predicts whether a track is explicit (`1`) or non-explicit (`0`). It leverages various audio characteristics such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, and valence to make the prediction.

## Features

- **danceability**: Suitability of the track for dancing.
- **energy**: Intensity and activity level of the track.
- **loudness**: Overall loudness in decibels (dB).
- **speechiness**: Presence of spoken words.
- **acousticness**: Confidence measure of whether the track is acoustic.
- **instrumentalness**: Prediction of whether a track contains no vocals.
- **liveness**: Presence of a live audience in the recording.
- **valence**: Musical positiveness conveyed by a track.

## Model Used

- **RandomForestClassifier**: An ensemble learning method that constructs multiple decision trees and outputs the mode of the classes.

## Process

1. **Data Preprocessing**: Load the dataset and select relevant features and the target variable. Convert the target variable from boolean to integer.
2. **Train-Test Split**: Split the data into training and testing sets.
3. **Model Training**: Initialize and train the RandomForestClassifier using the training data.
4. **Prediction**: Predict explicit content on the test data.
5. **Evaluation**: Evaluate the model using accuracy and a classification report (precision, recall, and F1-score).

## Evaluation Metrics

- **Accuracy**: Ratio of correctly predicted instances to the total instances.
- **Classification Report**: Includes precision, recall, and F1-score for both classes (explicit and non-explicit).
