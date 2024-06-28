import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader 

batch_size = 2

# Get data for inputs and labels
def get_data(data):
    inputs = torch.tensor(data[:-1])
    labels = torch.tensor(data[1:])
    return inputs, labels
    
# Sample Text 
corpus = """India is my country.
            Australia is a country."""

tokens = corpus.split()

# Generate vocabulary and a dictionary
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)
stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for i, s in enumerate(vocab)}

# Encoding and Decoding 
encode = lambda text: [stoi[word] for word in text.split()]
decode = lambda nums: ' '.join([itos[index] for index in nums])

data = encode(corpus)
inputs, labels = get_data(data)
inputs_one_hot = F.one_hot(inputs, num_classes=vocab_size).float()  # One hot encoding for input to the Embedding Layers
labels_one_hot = F.one_hot(labels, num_classes=vocab_size).float()  # One hot encoding for input to the Embedding Layers

dataset = TensorDataset(inputs_one_hot, labels_one_hot)
data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Word Embedding Class
class WordEmbeddingWithLinear(nn.Module):
    def __init__(self, vocab_size, embedding_dim=2):
        super(WordEmbeddingWithLinear, self).__init__()
        
        self.input_to_hidden = nn.Linear(in_features=vocab_size, out_features=embedding_dim, bias=False)
        self.hidden_to_output = nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
    def forward(self, inputs):
        hidden = self.input_to_hidden(inputs.float())
        output_values = self.hidden_to_output(hidden)
        return (output_values)
    
    def train_model(self, data, epochs=100):
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in data:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss/len(data):.4f}')

# Model to train
modelLinear = WordEmbeddingWithLinear(vocab_size)

# Plot embeddings on a graph to show the embedding space
def plot_embeddings(model, title):
    weights1 = model.input_to_hidden.weight.detach()[0].numpy()
    weights2 = model.input_to_hidden.weight.detach()[1].numpy()
    datas = {
        "w1": weights1,
        "w2": weights2,
        "token": [itos[i] for i in range(vocab_size)]
    }
    df = pd.DataFrame(datas)
    sns.scatterplot(data=df, x="w1", y="w2")
    for i in range(df.shape[0]):
        plt.text(df.w1[i], df.w2[i], df.token[i], 
                 horizontalalignment='left', size='medium', color='black', weight='semibold')
    plt.title(title)
    plt.show()

# Before training the model (Embedding Space)
plot_embeddings(modelLinear, "Initial Embeddings")

modelLinear.train_model(data)
weights1 = modelLinear.input_to_hidden.weight.detach()[0].numpy()
weights2 = modelLinear.input_to_hidden.weight.detach()[1].numpy()
datas = {
    "w1": weights1,
    "w2": weights2,
    "token": [itos[i] for i in range(vocab_size)]
}
df = pd.DataFrame(datas)
print(df)

# After training the model (Embedding Space)
plot_embeddings(modelLinear, "Embeddings After Training")
