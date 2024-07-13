
from flask import Flask, render_template, redirect, url_for, request




app = Flask(__name__)


@app.route("/")
def login():
    return render_template("index.html")

@app.route("/try")
def try1():
    return render_template("try.html")








if __name__=="__main__":
    app.run(debug=True)

