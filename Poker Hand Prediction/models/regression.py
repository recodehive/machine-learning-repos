'''
Multi-linear regression.
'''

# -------------------- Imports -------------------- #
from sklearn import linear_model as lm
import math
from poker_hand_prediction import *


# -------------------- Model -------------------- #
model = lm.LinearRegression()

# FIT (i.e train)
trainedModel = model.fit(train_x, train_y)

# TEST (i.e predict)
predictions = model.predict(test_x)

# Errors
avgError = sum([math.fabs(x-y) for x, y in zip(predictions, test_y)]) / len(predictions)
print("Average error :", avgError)

rmsError = sum([(x-y)**2 for x, y in zip(predictions, test_y)]) / len(predictions)
rmsError = math.sqrt(rmsError)
print("RMSE :", rmsError)

print("Model R2 score :", model.score(train_x, train_y))