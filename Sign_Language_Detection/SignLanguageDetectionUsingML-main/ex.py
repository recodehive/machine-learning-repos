from flask import Flask, render_template,redirect,url_for,request
import mysql.connector
import MySQLdb.cursors
import re

conn = mysql.connector.connect(host="localhost", user="root", password="Shraddha@2004", database="mydb")
print(conn)
c = conn.cursor()
# c.execute("SELECT * FROM LoginForm;")
# myresult = c.fetchall()
# for x in myresult:
#   print(x)
app = Flask(__name__)

@app.route("/form")
def form():
    return render_template("index.html")

@app.route("/in", methods=['POST'])
def formin():
    fname=request.form['fname']
    lname=request.form['lname']
    mname=request.form['mname']
    faname=request.form['faname']
    address=request.form['address']
    dob=request.form['dob']
    pincode=request.form['pincode']
    course=request.form['course']
    email=request.form['email']
    sql="INSERT INTO form VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s);"
    val=(fname,lname,mname,faname,address,dob,pincode,course,email)
    c.execute(sql,val)
    print(c)
    conn.commit()
    conn.close()
    return render_template("course.html")

if __name__=="__main__":
    app.run(debug=True)
