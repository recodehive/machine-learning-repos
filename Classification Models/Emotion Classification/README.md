# Emotion Classification from Text

This project implements an emotion classification model using a pre-trained transformer model (DistilBERT) to classify emotions based on text inputs. The model is trained on a dataset containing various emotional statements.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)

## Overview

The goal of this project is to classify emotions expressed in text using natural language processing (NLP) techniques. We leverage the Hugging Face Transformers library to fine-tune a pre-trained DistilBERT model on our dataset.

## Dataset

The dataset used for training the model should have the following structure:

| content                        | sentiment |
|--------------------------------|-----------|
| alonzo feels angry             | anger     |
| alonzo feels sad               | sadness   |
| alonzo feels terrified          | fear      |

Make sure to place your dataset in the project directory and name it `emotion_data.csv`.

## Installation

To run this project, you'll need to install the required Python packages. You can do this using pip:

```bash
pip install transformers torch pandas scikit-learn
