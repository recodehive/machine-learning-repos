from tkinter import *;
from tkinter import Tk,Toplevel,Listbox
from tkinter import filedialog
from PIL import Image, ImageTk
import SignLanguageDetectionUsingML-mainos
import sys
def restart_program():
 python = sys.executable
 os.execl(python, python, *sys.argv)
class Ask(Tk):
 def _init_(self):
 self.im=NONE
 self.open_window()
 def open_window(self):
 self.im = filedialog.askopenfilename(parent=root, title="select
image to compare")
class MyApp(Tk):
 def _init_(self,parent):
27
 Tk._init_(self,parent)
 self.parent=parent
 self.initialize()
 def initialize(self):
 top=Frame(master=None, bd=0,bg='#C6DEFF')
 top.pack(fill=X,padx=0,pady=0)
 bottom=Frame(Master=None, height=300, bd=0,bg='#C6DEFF')
 bottom.pack(fill=BOTH, padx=3, pady=3, expand=1)
 topleft=Frame(master=top, height=22, width=500, bd=0)
 topleft.pack(side=LEFT, fill=Y,padx=2, pady=2)
 topright=Frame(master=top, height=100, width=500,
bd=1,bg='#C6DEFF')
 topright.pack(fill=Y, padx=4, pady=20)
 def callback():
 r_value()
 can_height()
 j=groupProject.MyImage(a[43:],1)
 list1=j.distancewith[1:]
 self.load_images(self.can, list1,self.r)
 print("Done!")

 def callback2():
 restart_program()
28
 restart=Button(master=topright,text="Browse",command=
lambda: callback2(), font="Calibri 14 bold",bg='#C2DFFF')
 restart.pack(side=RIGHT, anchor=SE,padx=5, pady=5)
 retrieve=Button(master=topright, text="Retrieve SImilar
Images",command=callback,font="Calibri 14 bold",bg='#C2DFFF')
 retrieve.pack(side=RIGHT, anchor=SE, padx=5, pady=5)
 var=str(win1.im)
 var=var[43:]
 quote="''"
 quote=str(quote)
 myphot_title=Label(topleft,text="User Iaage Input:" + quote + var
+ quote )
 myphot_title.pack(anchor='nw')
 topleftbot=Frame(master=topleft,bd=5,relief=SUNKEN)
 topleftbot.pack(side=BOTTOM,padx=4,pady=4)
 can1=Canvas(master=topleftbot, width=box[2]-2, height=box[3]-
2)
 can1.pack(fill=BOTH)
 can1photo=ImageTk.PhotoImage(myphoto_thumb)
 self.can1photo=can1photo
 can1.create_image(0,0, image=can1photo,anchor=NW)
 method_ask="Choose number of images you want to retrieve"
 Label(topright, text=method_ask,font="Calibri
12",bg='#C6DEFF').pack(anchor=W)
29

 self.r=10
 def r_value():
 if w.get()==2:
 self.r=20
 elif w.get()==3:
 self.r=25
 else:
 self.r=10
 w=IntVar()
 opt1="10"
 opt2="20"
 opt3="25"

red3=Radiobutton(master=topright,text=opt1,variable=w,value=1,font=
"Calibri 12",bg='#C6DEFF')

red4=Radiobutton(master=topright,text=opt2,variable=w,value=2,font=
"Calibri 12",bg='#C6DEFF')

red5=Radiobutton(master=topright,text=opt3,variable=w,value=3,font=
"Calibri 12",bg='#C6DEFF')
 red3.pack(anchor=W)
 red4.pack(anchor=W)
 red5.pack(anchor=W)
 red3.select()
 Label(bottom,text="Similar Images").pack(side=TOP,anchor=W)
 scroll_h=250
30
 def can_height():
 if w.get()==1 or w.get()==0:
 scroll_h=250
 if w.get()==2:
 scroll_h=500
 if w.get()==3:
 scroll_h=1000
 self.can.config(scrollregion=(0,0, scroll_h,scroll_h))
 can_width=(5*size[0])+(7*1)
 self.bottombt=Frame(master=bottom,height=78, width=100,
bd=1, relief=SUNKEN)
 self.bottombt.pack(side=BOTTOM,fill=BOTH,padx=2,
pady=2,expand=1)
 self.can=Canvas(self.bottombt,bg='#FFFFFF',width=can_width,
scrollregion=(0,0, scroll_h,scroll_h))
 scrollbar=Scrollbar(self.bottombt)
 scrollbar.pack(side=RIGHT,fill=Y)
 scrollbar.config(command=self.can.yview)
 self.can.config(yscrollcommand=scrollbar.set)
 self.can.pack(expand=True, fill=BOTH)
 self.load_images(self.can, list1, self.r)
 def load_images(self, canvas, list1, how_many):
 countx=0
 county=0
31
 r=how_many
 l=[0]*r
 for i in range(r):
 p=Image.open(list1[i][1])
 p_thumb=p.copy()
 p_thumb.thumbnail(size)
 l[i]=["photo{0}".format(i)]
 l[i]=ImageTk.PhotoImage(p_thumb)
 self.l=l
 x=1+(countx*size[0])+(countx*1)
 y=1+(county*size[1])+(county*1)
 canvas.create_image(x,y, image=l[i], anchor=NW)
 if countx==4:
 countx=0
 county+=1
 else:
 countx+=1
 return
if _name_ == '_main_':
 root = Tk()
 root.withdraw()
 root.config(bg='#D7F5F7')
 win1 = Ask()
 root.title("Content Based Image Retrival System")
 size = 170, 170
 size2 = 328, 328
32
 myphoto = Image.open(win1.im)
 a = str(win1.im)
 i = groupProject.MyImage(a[43:], 1)
 list1 = i.distancewith[1:]
 myphoto_thumb = myphoto.copy()
 myphoto_thumb.thumbnail(size2)
 box = myphoto_thumb.getbbox()
 root.deiconify()
 app = MyApp(None)
 app.title("Content Based Image Retrieval System")
 app.update()
 app.destroy()
 app.mainloop()
33
5.5. O