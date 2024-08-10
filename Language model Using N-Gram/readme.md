# N-gram Language Model for Sentence Auto-Completion

This repository contains the implementation of a language model designed for sentence auto-completion using the N-gram model. The project includes various preprocessing techniques such as tokenization, removing punctuation, and lowercasing to prepare the text data.

## Table of Contents
- [Introduction](#introduction)
- [Preprocessing Techniques](#preprocessing-techniques)
  - [Tokenization](#tokenization)
  - [Removing Punctuation](#removing-punctuation)
  - [Lowercasing](#lowercasing)
- [N-gram Model](#n-gram-model)
  - [Mathematical Explanation](#mathematical-explanation)
  - [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)

## Introduction

The goal of this project is to develop a language model that can automatically complete sentences based on a given context using the N-gram technique. This model is a foundational component for various natural language processing (NLP) tasks, such as predictive text and autocomplete features in text editors.

## Preprocessing Techniques

To ensure that the text data is clean and consistent, the following preprocessing techniques are applied:

### Tokenization

Tokenization is the process of splitting a sentence or text into individual words or tokens. This is a crucial step in the preparation of text data for the N-gram model.

### Removing Punctuation

Punctuation marks are removed from the text to reduce noise and ensure that the model focuses only on the words themselves.

### Lowercasing

All text is converted to lowercase to maintain consistency and reduce the complexity of the language model.

## N-gram Model

### Mathematical Explanation

The N-gram model is a probabilistic language model that predicts the next word in a sequence based on the previous $n-1$ words. The probability of a word $w$ given its previous words is computed as:

$$
P(w_n | w_1, w_2, \dots, w_{n-1}) = \frac{\text{Count}(w_{n-1}, w_n)}{\text{Count}(w_{n-1})}
$$

Where:
- $P(w_n | w_1, w_2, \dots, w_{n-1})$ is the probability of word $w_n$ given the previous $n-1$ words.
- $\text{Count}(w_{n-1}, w_n)$ is the frequency count of the sequence of the previous word and the current word.
- $\text{Count}(w_{n-1})$ is the frequency count of the previous word.

### Implementation Details

The model is implemented using Python and various NLP libraries. The steps include:
1. Preprocessing the input text using the techniques mentioned above.
2. Building the N-gram model by counting the frequency of word sequences.
3. Using the model to predict the next word in a given context for sentence auto-completion.

## Usage

To use this N-gram model for sentence auto-completion, follow these steps:

1. Clone the repository.
   ```bash
   git clone https://github.com/recodehive/machine-learning-repos/Language model Using N-Gram.git
   
2. Ensure you have the required dependencies installed (e.g., `nltk` for text processing).
3. Run the Jupyter notebook provided in the repository to train the model and test it on sample sentences.

## Results

The model successfully predicts the next word in a sentence based on the context provided by the preceding words. The accuracy of the predictions depends on the size of the N-gram and the quality of the training data.

## Limitations

- **Data Dependency**: The model's performance is highly dependent on the size and quality of the training corpus.
- **Limited Context**: The N-gram model only considers a fixed number of preceding words, which may lead to less accurate predictions for longer contexts.
- **Sparse Data**: For higher values of $n$, the model may encounter sparsity issues, where certain word combinations are rare or unseen in the training data.