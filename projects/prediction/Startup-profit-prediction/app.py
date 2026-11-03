import os
import pickle
import numpy as np
from flask import Flask, redirect, render_template, url_for, request

# Safely resolve model file path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'startup.pkl')
with open(MODEL_PATH, 'rb') as f:
    regressor = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == "POST":
        state = request.form.get("state", "California")
        rdspend = float(request.form.get("rdspend", 0.0))
        adspend = float(request.form.get("adspend", 0.0))
        mkspend = float(request.form.get("mkspend", 0.0))

        if state == "New York":
            state_list = [0.0, 1.0]
        elif state == "California":
            state_list = [0.0, 0.0]
        else:
            state_list = [1.0, 0.0]

        features = np.array(state_list + [rdspend, adspend, mkspend], dtype=float)
        features = features.reshape(1, -1)
        pred = regressor.predict(features)[0]

        return render_template("output.html", pred=pred)

    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)
