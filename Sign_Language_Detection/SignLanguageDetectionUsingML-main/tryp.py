from tkinter import *
from function import *
import subprocess

window =Tk()

window.title("Sign Language Detection")
window.geometry("500x500")


# Create a button widget
button = Button(window, text="For Alphabets", command=lambda:subprocess.run(["python","gui.py"]))
button.place(x=200,y=250)


window.mainloop()