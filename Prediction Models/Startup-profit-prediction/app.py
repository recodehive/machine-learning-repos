from flask import Flask, redirect, render_template, url_for, request
import numpy as np
import pickle

regressor = pickle.load(open('startup.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == "POST":
        state = request.form["state"]
        rdspend = float(request.form["rdspend"])
        adspend = float(request.form["adspend"])
        mkspend = float(request.form["mkspend"])
        if state == "New York":
            state_list = [0.0, 1.0]
        elif state == "California":
            state_list = [0.0, 0.0]
        else:
            state_list = [1.0, 0.0]

        input = np.array(state_list+[rdspend, adspend, mkspend])
        input = input.reshape(1, len(input))
        pred = regressor.predict(input)[0]

    return render_template("output.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True)
