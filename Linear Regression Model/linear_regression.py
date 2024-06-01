import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("/kaggle/input/salary-dataset-simple-linear-regression/Salary_dataset.csv")

plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot")
plt.show()

X=df[['YearsExperience']]
y=df['Salary']

model=LinearRegression()
model.fit(X,y)  #calculates best fitting relation between x and y
slope = model.coef_[0]  # represents coeff of x in y = mx + b
intercept = model.intercept_  # point where regression lines crosses y axis

def linear_regression(x):
    return slope * x + intercept

plt.scatter(df['YearsExperience'],df['Salary'], label='Data Points')
plt.plot(df['YearsExperience'], [linear_regression(xi) for xi in df['YearsExperience']], color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Linear Regression")
plt.show()

new_x = 6
prediction = linear_regression(new_x)
print(f"Prediction for x = {new_x}: {prediction}")

new_x = pd.DataFrame({"YearsExperience": [6]})
print(model.predict(new_x))


new_x = pd.DataFrame({'YearsExperience': [0]})
print(model.predict(new_x))
