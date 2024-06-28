# Word Embedding with Linear Model

This project demonstrates a simple neural network model to create word embeddings using a linear transformation. The model is implemented using PyTorch and trained on a small corpus of text.

## Prerequisites

Ensure you have the following libraries installed:

- `torch`
- `pandas`
- `seaborn`
- `matplotlib`

You can install these libraries using pip:

```bash
pip install torch pandas seaborn matplotlib
```

## Usage

1. **Prepare the Data:**
   The text corpus is tokenized, and a vocabulary is created. Each word is then one-hot encoded.

2. **Define the Model:**
   The `WordEmbeddingWithLinear` class defines a simple linear model with an input-to-hidden and a hidden-to-output linear layer.

3. **Train the Model:**
   The model is trained using the provided data loader.

4. **Plot the Embeddings:**
   The embeddings are visualized using `seaborn` and `matplotlib` before and after training.

## Results

The model's embeddings for each word in the vocabulary are plotted before and after training, showing how the words are distributed in the embedding space.

## Notes

- Ensure your text corpus is large enough to provide meaningful embeddings.
- Experiment with different embedding dimensions and learning rates to see how they affect the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses code snippets and techniques from various tutorials and documentation available in the PyTorch community.

---

This README provides a clear and concise overview of the project, prerequisites, usage steps, and additional notes without including the code itself.
