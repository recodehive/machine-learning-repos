'''
Deep Neural Net.
'''

# -------------------- Imports -------------------- #
from keras.models import Sequential
from keras.layers import Dense

from poker_hand_prediction import *


# -------------------- Preparign the Data -------------------- #
train_y_onehot = list()
for y in range(len(train_y)):
    temp = [0] * config.classes
    temp[train_y[y]] = 1
    train_y_onehot.append(temp)
    
test_y_onehot = list()
for y in range(len(test_y)):
    temp = [0] * config.classes
    temp[test_y[y]] = 1
    test_y_onehot.append(temp)
    
train_y_onehot = np.array(train_y_onehot)
test_y_onehot = np.array(test_y_onehot)


# -------------------- Model -------------------- #
model = Sequential()

# Input layer
model.add(Dense(config.features, input_shape = (train_x.shape[1],), activation='tanh'))

# Hidden Layers
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))

# Output layer
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y_onehot, epochs = 500, batch_size = 500, verbose=0)

scores = model.evaluate(train_x, train_y)
print("Train =", model.metrics_names[1], scores[1] * 100)

scores = model.evaluate(test_x, test_y)
print("Test =", model.metrics_names[1], scores[1] * 100)