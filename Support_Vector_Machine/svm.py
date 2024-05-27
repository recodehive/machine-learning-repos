# Import Libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Load the Breast Cancer dataset
cancer_data = datasets.load_breast_cancer()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    cancer_data.data, cancer_data.target, test_size=0.4, random_state=111
)

# Create a SVM classifier with a linear kernel
data = svm.SVC(kernel="linear")

# Train the classifier
data.fit(x_train, y_train)

# Make predictions on the test set
pred = data.predict(x_test)

# Evaluate the model's accuracy
print("acuracy:", (metrics.accuracy_score(y_test, y_pred=pred)) * 100)
