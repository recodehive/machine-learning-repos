import json

from torch._C import device
from torch.nn.modules import loss
from nltk_utils import stem,bag_of_words,tokenize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetModels



with open('intents.json','r') as f:
    intents=json.load(f)

#print(intents)
all_words=[]
tags=[]
tag_pattern_words=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)  #didnt use append cuz w is array itself and we dont want array of array in all_words
        tag_pattern_words.append((w,tag))

ignore_words=['?,','.',',','!']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))  #using set to remove duplicate words
tags=sorted(set(tags))
#print(tags)

X_train=[]
Y_train=[]

for pattern,tag in tag_pattern_words:
    bag=bag_of_words(pattern,all_words)
    X_train.append(bag)

    label=tags.index(tag)
    Y_train.append(label)  #CrossEntropy Loss

X_train=np.array(X_train)
Y_train=np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    
    def __getitem__(self, idx) :
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples
    
#Hyperparameters
batch_size = 32
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])   #or you can say len(all_words)
learning_rate = 0.001
num_epochs = 100000
#print(input_size,len(all_words))
#print(output_size,tags)


dataset = ChatDataset()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True, num_workers=0)
model = NeuralNetModels(input_size,hidden_size,output_size).to(device)
#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)




for epoch in range(num_epochs) :
    for (words, labels)in train_loader:
        words = words.to(device)
        
        labels = labels.to(dtype=torch.long)
        labels = labels.to(device)
        #call fwd pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step

        optimizer.zero_grad()
        loss.backward()  #to call backward pro
        optimizer.step()

    if (epoch+1)%100 == 0:
        print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}')


print(f'final loss, loss={loss.item():.4f}')
