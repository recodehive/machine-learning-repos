# Part-of-Speech Tagging Using Hidden Markov Model (HMM)

This project provides a Python class `PosTagging` for performing Part-of-Speech (POS) tagging using a Hidden Markov Model (HMM). The class trains on a set of tagged sentences and uses the learned model to tag new sentences.

## Table of Contents
- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Implementation Details](#implementation-details)
- [Advantages and Limitations](#advantages-and-limitations)
- [License](#license)

## Introduction

POS tagging is a fundamental task in Natural Language Processing (NLP) where each word in a sentence is labeled with its corresponding part of speech (e.g., noun, verb, adjective). This implementation uses an HMM-based approach to assign the most likely tag sequence to a sentence.

## How It Works

The `PosTagging` class is designed to:

1. **Train**: It calculates the transition and emission probabilities based on a provided training dataset of tagged sentences.
2. **Tag**: It assigns the most likely POS tags to each word in an unseen sentence using the Viterbi-like algorithm based on the learned probabilities.

## Implementation Details

### Transition and Emission Probabilities

- **Transition Probability**: The probability of a tag $T_i$ given the previous tag $T_{i-1}$. This is calculated during the training phase by counting tag sequences.
  
$$
  P(T_i | T_{i-1}) = \frac{\text{Count}(T_{i-1}, T_i)}{\sum_{T_j} \text{Count}(T_{i-1}, T_j)}
 $$

- **Emission Probability**: The probability of a word $W_i$ given a tag $T_i$. This is calculated based on how often a word is associated with a tag in the training data.
  
$$
  P(W_i | T_i) = \frac{\text{Count}(T_i, W_i)}{\sum_{W_j} \text{Count}(T_i, W_j)}
 $$

### Tagging Algorithm

For each word in the input sentence, the class calculates the product of the transition and emission probabilities for each possible tag and assigns the tag with the highest probability.

### Key Attributes

- **`transition`**: A dictionary storing the transition probabilities between tags.
- **`emission`**: A dictionary storing the emission probabilities for each word-tag pair.
- **`tag_set`**: A set containing all the unique tags in the training data.
- **`word_set`**: A set containing all the unique words in the training data.

## Advantages and Limitations

### Advantages
- **Simplicity**: The HMM-based approach is straightforward and interpretable.
- **Efficiency**: The algorithm efficiently computes the most likely tag sequence for a sentence.

### Limitations
- **Sparsity**: The model may struggle with unseen words or tags not present in the training data.
- **Context**: HMMs assume that the tag of a word depends only on the previous tag, which can be limiting for capturing long-range dependencies.
