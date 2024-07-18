'''
Output Code Classifer
'''

# -------------------- Imports -------------------- #

from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from poker_hand_prediction import *

# -------------------- Model -------------------- #

print("\nOutput Code Classifer")

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

occ = OutputCodeClassifier(BaggingClassifier())
occ.fit(data_train, train_y)
prediction = occ.predict(data_test)
accuracy = accuracy_score(test_y, prediction)

# -------------------- Print the final result -------------------- #

print("\nAccuracy using Output Code Classifier :", round(accuracy * 100, 3),"%\n")
