'''
SVMs for multi-class classification.
'''

# -------------------- Imports -------------------- #
import sys
from sklearn import svm
from poker_hand_prediction import *


# -------------------- Model -------------------- #
model = svm.SVC()
print("Generated model! Going to train...")

# Training
model.fit(train_x, train_y)
print("Done training! Going to test...")

# Testing

predictions = list()
for l in range(len(test_x)):
    predictions.append(model.predict([test_x[i]])[0])

print("Generated predictions!")