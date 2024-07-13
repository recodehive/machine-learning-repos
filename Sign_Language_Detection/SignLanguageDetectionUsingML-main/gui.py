from tkinter import *
from tkinter import *
from PIL import ImageTk, Image
import subprocess


# Create the Tkinter window
window =Tk()
window.title("Sign Language Detection")
window.geometry("800x600")


image = Image.open("backbg.jpg")  # Replace "path_to_image_file.jpg" with the actual path to your image file
image = image.resize((window.winfo_screenwidth(), window.winfo_screenheight()), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image)

background_label = Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
# Create a button widget

button = Button(window, padx=40, pady=20,text="For Alphabets", bg='#CAEBF0',command=lambda: subprocess.run(["python", "app.py"]))
button .place(x=200,y=250)

button1 =Button(window,padx=40, pady=20, text="For Words", bg='#CAEBF0', command=lambda: subprocess.run(["python", "words.py"]))
button1 .place(x=400,y=250)



# Start the Tkinter event loop
window.mainloop()

