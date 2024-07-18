'''
Multi-layer perceptron with 2 hidden layers and stochastic gradient descent 
'''

# -------------------- Imports -------------------- #
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from poker_hand_prediction import *

# -------------------- Model -------------------- #

print("\nMulti Layer Perceptron with 2 hidden layers and stochastic gradient descent")

# Initialise scaler to scale the data
scaler = StandardScaler()

train_set = np.empty(train_x.shape, dtype = float)
test_set = np.empty(test_x.shape, dtype = float)

# Make training data suitable for scaling
for index in range(len(train_x)):
    train_set[index] = train_x[index].astype(float)

# Make testing data suitable for scaling
for index in range(len(test_x)):
    test_set[index] = test_x[index].astype(float)

# Fit the training data
scaler.fit(train_set)  

# Scale the training and testing data w.r.t scaler
data_train = scaler.transform(train_set)
data_test = scaler.transform(test_set)

# Store the accuracies
accuracy = list()

for trial in range(5):
    classifier = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (64, 64), activation = 'tanh', learning_rate_init = 0.02, max_iter = 2000, random_state = trial)
    result = classifier.fit(data_train, train_y)
    prediction = classifier.predict(data_test)
    curr_accuracy = accuracy_score(test_y, prediction)
    accuracy.append(curr_accuracy)

# -------------------- Print the final result -------------------- #

tab = PrettyTable(['Trial Number', 'Accuracy'])

for trial in range(len(accuracy)):
    tab.add_row([trial + 1, round(accuracy[trial] * 100, 3)])

print("\n", tab)
print("\nMean Accuracy using Multi Layer Perceptron Classifier: ", round(np.array(accuracy).mean() * 100, 3), "%")
