from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
import pickle

# laoding model
model = pickle.load(open('model.pkl','rb'))
# flask app
app = Flask(__name__)

# paths
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
   age  = request.form['age']
   operational_year     = request.form['operational_year']
   exil_node = request.form['exil_node']

   features = np.array([[age,operational_year,exil_node]])
   features = sclr.fit_transform(features)
   pred = model.predict(features).reshape(1,-1)

   return render_template('index.html', output = pred[0])

# python main
if __name__ == "__main__":
    app.run(debug=True)
