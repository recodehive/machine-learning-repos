# Sarcasm Detection Using NLP

This repository contains various deep learning models for detecting sarcasm in text data using TensorFlow and Keras. Each model explores different architectures and techniques to improve sarcasm detection performance.

## Dataset

The models are designed to work with any text classification dataset. Ensure your dataset is preprocessed and tokenized before feeding it into the models.


## Models

1. **Model 1: Global Average Pooling**
   - Embedding layer followed by a GlobalAveragePooling1D layer and Dense layers.
   - Simple and efficient for basic text classification tasks.

<img width="534" alt="model1_acc" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/545b7b2c-4a12-404c-afb9-8fdd04f91748">
<img width="552" alt="model1_loss" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/eb148d69-7739-4852-8ba5-58816a18278e">


2. **Model 2: Stacked Bidirectional LSTM**
   - Two Bidirectional LSTM layers stacked together.
   - Suitable for capturing long-term dependencies in the text data.

<img width="567" alt="model2_acc" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/eba6fe0c-94d1-4920-87d2-b9bea09d5b6b">
<img width="522" alt="model2_loss" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/f1c8c30f-87fa-42c1-892b-f40e9f37fc99">

3. **Model 3: Single Bidirectional LSTM**
   - A single Bidirectional LSTM layer with Dense layers.
   - Effective for capturing sequential information with lower complexity.

<img width="537" alt="model3_acc" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/954623ed-5618-483a-a684-bd70d8117ae8">
<img width="528" alt="model3_loss" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/dff08bf1-851c-4129-acf2-6723c465f687">

4. **Model 4: Conv1D + Bidirectional LSTM**
   - Convolutional layers followed by a Bidirectional LSTM.
   - Combines local feature extraction with sequence modeling.

<img width="537" alt="model4_acc" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/9c71932a-530e-4c18-8779-498a489fd094">
<img width="516" alt="model4_loss" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/9232d1ac-8fd1-4567-9c17-0607b310fbcf">

5. **Model 5: CNN-LSTM Hybrid with Batch Normalization**
   - Combines Conv1D and Bidirectional LSTM layers with Batch Normalization.
   - Enhances feature extraction and improves convergence.

<img width="555" alt="model5_acc" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/100e3005-cfaf-4045-8707-a00fab04af88">
<img width="525" alt="model5_loss" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/8bdc736e-0b40-4f4a-b063-1c436a3059e1">

6. **Model 6: GRU-Based Model with Dropout**
   - Uses GRU layers with Dropout for regularization.
   - Potentially faster training with regularization to prevent overfitting.

<img width="561" alt="model6_acc" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/21bb8bcc-5c3f-4e35-a75d-fa53f314c7c9">
<img width="528" alt="model6_loss" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/1cc763b0-c020-4cf3-bfcb-acb3edfdbaac">

7. **Model 7: Attention Mechanism**
   - Incorporates an attention layer after Bidirectional LSTM.
   - Helps the model focus on important parts of the input sequence.

<img width="569" alt="model7_acc" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/bf58e611-e4c0-44e4-bfa6-700d21fa2bc7">

<img width="523" alt="model7_loss" src="https://github.com/KamakshiOjha/ML-Crate/assets/114620432/ca5b409f-6f2c-4fb6-837a-38793e76532c">

## Conclusion

Experiment with different models and architectures provided in this repository to find the best performing model for your text classification task. Each model offers unique strengths, and their performance may vary depending on your specific dataset and task requirements.
