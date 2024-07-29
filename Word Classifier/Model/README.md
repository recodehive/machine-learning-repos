# Model
  The model is based on the transfer learning from Tensorflow's [NNLM Model](https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2).
  It has two hidden layers apart from the `base model`, and the output layer with `softmax` activation.

  The dataset is custom made with the help of internet to get various transcripts. The model reaches 2% validation loss and
  84% validation accuracy. But, the dataset is too small and can be modified to improve.

  **IF THE MODEL FAILS TO DOWNLOAD UPON INITIAL LAUNCH, REFER TO [HERE](Model_Data/README.md)**.
