# Neural Machine Translation (NMT)-Based Autocorrect Model

This project is an implementation of an autocorrect system using a sequence-to-sequence neural machine translation (NMT) model with attention. The model is trained to correct spelling mistakes in sentences by learning to translate noisy text into correct English sentences.

## Introduction
The autocorrect model is built using a sequence-to-sequence (Seq2Seq) approach with an encoder-decoder architecture. Bahdanau attention is incorporated to help the model focus on relevant parts of the input sentence during decoding. This model aims to correct noisy or misspelled sentences by "translating" them into their corrected versions.

## Model Architecture

The model consists of the following components:
- **Encoder**: Processes the input sequence and outputs the hidden states.
- **Bahdanau Attention**: Calculates attention weights for each input timestep, helping the decoder focus on relevant parts of the input.
- **Decoder**: Uses the encoder's context vectors and its own hidden states to generate the corrected output sequence.

### Custom Objects
- `Encoder`: Encodes the input sequence into context vectors.
- `BahdanauAttention`: Computes attention weights to focus on important parts of the input.
- `Decoder`: Generates the output sequence by attending to the encoder's context vectors.

## Known Issues and Fixes
- The dataset is large and takes a lot of time to train. As a result, only a portion of the data can be used to train the model.
- This model sometimes repeats words in a sentence.
- A system with high specs can be used to compile a model with a much larger portion of the downloaded dataset to ensure higher accuracy and precision.

## Dependencies
Install the following dependencies:
```bash
pip install tensorflow==2.13.0
pip install numpy
pip install pandas
