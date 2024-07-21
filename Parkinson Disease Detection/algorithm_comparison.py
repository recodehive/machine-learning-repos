# import all necessary libraries
import pandas
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load the dataset (local path)
url = "data.csv"
# feature names
features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE","status"]
dataset = pandas.read_csv(url, names = features)

# store the dataset as an array for easier processing
array = dataset.values
# X stores feature values
X = array[:,0:22]
# Y stores "answers", the flower species / class (every row, 4th column)
Y = array[:,22]
validation_size = 0.3
# randomize which part of the data is training and which part is validation
seed = 7
# split dataset into training set (80%) and validation set (20%)
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size = validation_size, random_state = seed)

# 10-fold cross validation to estimate accuracy (split data into 10 parts; use 9 parts to train and 1 for test)
num_folds = 10
num_instances = len(X_train)
seed = 7
# use the 'accuracy' metric to evaluate models (correct / total)
scoring = 'accuracy'

# algorithms / models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NN', MLPClassifier(solver='lbfgs')))
models.append(('NB', GaussianNB()))
models.append(('GB', GradientBoostingClassifier(n_estimators=10000)))

# evaluate each algorithm / model
results = []
names = []
print("Scores for each algorithm:")
for name, model in models:
    kfold = cross_validation.KFold(n = num_instances, n_folds = num_folds, random_state = seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name, accuracy_score(Y_validation, predictions)*100)
    print(matthews_corrcoef(Y_validation, predictions))
    print()
