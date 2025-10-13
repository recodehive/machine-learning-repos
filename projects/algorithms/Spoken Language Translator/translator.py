from tkinter import *
from googletrans import Translator
import win32com.client as wincl
import speech_recognition as sr
class translator:

    def translate(self):
        if str(self.fn.get())!="":
            self.translatedfinal = self.translator.translate(str(self.fn.get()), dest=self.variable.get())
        else:
            self.translatedfinal = self.translator.translate(self.text, dest=self.variable.get())

        self.name = Label(self.root, text="Translated Text:", bg='black',fg='cyan',font='1')
        self.name.place(x=100, y=400)
        self.final = Label(self.root, text=self.translatedfinal.text+"                 ", font=100,bg='black',fg='cyan')
        self.final.place(x=400, y=400)


    def __init__(self):
        self.root = Tk()
        self.root.geometry('900x700')
        self.root.title("Translator")
        self.root.config(bg='black')
        self.speak = wincl.Dispatch("SAPI.SpVoice")

        self.raw = ""
        self.fn = StringVar()
        self.ln = StringVar()
        self.translator = Translator()

        self.languages = [
            "English",
            "Hindi",
            "Telugu",
            "Tamil",
            "Kannada",
            "Malayalam",
            "Marathi",
            "Gujarati",
            "Bengali",
            "Punjabi",
            "Odia",
            "Nepali",
            "Sindhi",
            "Sanskrit",
            "Russian",
            "French",
            "Arabic",
            "Bulgarian",
            "Danish",
            "German",
            "Greek",
            "Persian",
            "Italian",
            "Japanese",
            "Korean",
            "Polish",
            "Urdu",
            "Chinese",
            "Dutch",
            "Spanish",
            "Portuguese",
            "Romanian",
            "Swedish",
            "Turkish",
            "Vietnamese",
            "Afrikaans"            
        ]


        self.name = Label(self.root, text="Language Translator", bg='black',fg='cyan')
        self.name.config(font=("Old Stamper",30))
        self.name.place(x=200, y=130)

        self.input = Label(self.root, text="Enter text here:", bg='black',fg='cyan',font=100)
        self.input.place(x=100, y=305)

        self.firste = Entry(self.root, textvariable=self.fn, bg='black', fg='cyan',font='10')
        self.firste.place(x=280, y=305)


        self.trans = Button(self.root, text="Translate", bd=7, bg='black', fg='cyan',command=self.translate)
        self.trans.place(x=600, y=500)

        self.variable = StringVar(self.root)
        self.variable.set(self.languages[0])

        w=OptionMenu(self.root,self.variable,*self.languages)
        w.config(bg='black',fg='cyan',border=0)
        w.place(x=600,y=305)

        self.root.mainloop()

s = translator()
