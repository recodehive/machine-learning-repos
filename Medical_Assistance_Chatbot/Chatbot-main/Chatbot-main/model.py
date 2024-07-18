import torch
import torch.nn as nn

class NeuralNetModels(nn.Module) :
    def __init__(self, input_size,hidden_size,num_classes):
        super(NeuralNetModels,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)   #1st Layer with input size and output as hidden size
        self.l2 = nn.Linear(hidden_size,hidden_size)  #2nd layer with hidden size as input and hidden size as output
        self.l3 = nn.Linear(hidden_size,num_classes)  #3rd layer with hidden size as input and num of classes as output
        self.relu = nn.ReLU()                         #Activation 

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        return out