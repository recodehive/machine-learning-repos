from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRectangleFlatButton
from kivy.lang import Builder
from kivy.core.window import Window
import random
import joblib
import requests
from bs4 import BeautifulSoup

#Window.size=(320,600)
username_input="""
MDTextField:
    
    hint_text: "Enter city of locating "
    hint_text_size:30
    helper_text: "check the spelling"
    helper_text_mode: "on_focus"
    icon_right: "language-python"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.8}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'15sp'
    width:400
"""
username_input1 = """
MDTextField:
    hint_text: "Enter Gender 1-Male 0-Female:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "android"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.7}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'15sp'
    width:400
"""
username_input2 = """
MDTextField:
    hint_text: "Enter Actual PEFR value:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "android"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.6}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'15sp'
    width:400
"""
def predicter(city,g,actual_pefr):
    l=['Cuddalore','Tiruppur','Ooty','Krishnagiri','Kattivakkam','Dindigul','Chennai','Arcot','Ariyalur']
    model = joblib.load('decision_tree_model.joblib')
    url = f'https://www.iqair.com/in-en/india/tamil-nadu/{city}'
    r = requests.get(url)
    soup = BeautifulSoup(r.content,'html.parser')
    aqi_dict = []
    s = soup.find('table',class_ = "aqi-overview-detail__other-pollution-table")
    if(s==None):
        city = random.choice(l).lower()
        url = 'https://www.iqair.com/in-en/india/tamil-nadu/'+city
        r = requests.get(url)
        soup = BeautifulSoup(r.content,'html.parser')
        aqi_dict = []
        s = soup.find('table',class_ = "aqi-overview-detail__other-pollution-table")
    for x in s:
        aqi_dict.append(x.text)
    aqi = aqi_dict[1]
    a=aqi.split(" ")
    pm2_index = a.index("PM2.5")
    pm2 = a[pm2_index +1][0:2]
    if 'PM10' not in aqi_dict :
        pm10 = 1.38 * float(pm2)
    else:
        pm10_index = a.index("PM10")
        pm10 = a[pm10_index+ 1][0:2]
    t = soup.find('div', class_="weather__detail")
    y = t.text
    temp_index = y.find('Temperature')+11
    degree_index = y.find('°')
    temp = y[temp_index : degree_index]
    hum_index = y.find('Humidity')+8
    perc_index = y.find('%')
    hum = y[hum_index:perc_index]
    p=temp
    q=hum
    r=pm2
    s=pm10
    prediction = model.predict([[g,p,q,r,s]])
    predicted_pefr = prediction[0]
    perpefr = (actual_pefr/predicted_pefr)*100
    if perpefr >= 80:
        re='SAFE'
    elif perpefr >= 50:
        re='MODERATE'
    else:
        re='RISK'
    return (re,predicted_pefr,actual_pefr,(perpefr//100)*10)
class DemoApp(MDApp):
    
    def build(self):
        self.screen=Screen()
        self.theme_cls.primary_palette = "Green"
        self.label = MDLabel(text="ASTHMA RISK PREDICTION", halign="center",theme_text_color='Custom',
                        text_color=(0,1,0,1),font_style='H4',pos_hint={'center_x': 0.5, 'center_y': 0.9})
        self.username = Builder.load_string(username_input)
        self.username1 = Builder.load_string(username_input1)
        self.username2 = Builder.load_string(username_input2)
        
        self.btn = MDRectangleFlatButton(text='Calculate',font_size='20sp',
                                       pos_hint={'center_x': 0.5, 'center_y': 0.1},size_hint=(.2, .1)
                                             ,on_release=self.mul)
        self.screen.add_widget(self.label)
        self.screen.add_widget(self.username)
        self.screen.add_widget(self.username1)
        self.screen.add_widget(self.username2)
        self.screen.add_widget(self.btn)
        return self.screen
    def mul(self,obj):
        self.screen.remove_widget(self.label)
        self.screen.remove_widget(self.username)
        self.screen.remove_widget(self.username1)
        self.screen.remove_widget(self.username2)
        self.screen.remove_widget(self.btn)
        r=predicter(str(self.username.text),
                  int(self.username1.text),
                  int(self.username2.text),)
        self.label1 = MDLabel(text="PREDICTION RESULTS", halign="center",theme_text_color='Custom',
                        text_color=(0,1,0,1),font_style='H4',pos_hint={'center_x': 0.5, 'center_y': 0.9})
        self.label_1 = MDLabel(text="PREDICTED PEFR:   "+str(r[1]), halign="center",theme_text_color='Custom',
                        text_color=(0,0,1,1),font_style='H5',pos_hint={'center_x': 0.5, 'center_y': 0.7})
        self.label1_ = MDLabel(text="ACTUAL PEFR ENTERED:   "+str(r[2]), halign="center",theme_text_color='Custom',
                        text_color=(1,0,1,1),font_style='H5',pos_hint={'center_x': 0.5, 'center_y': 0.5}) 
        if(r[0]=='SAFE'):
            c=(0,1,0,1)
        if(r[0]=='MODERATE'):
            c=(1,1,0,1)
        if(r[0]=='RISK'):
            c=(1,0,0,1)
        self.label2 = MDLabel(text=str(r[0]), halign="center",theme_text_color='Custom',
                        text_color=c,font_style='H4',pos_hint={'center_x': 0.5, 'center_y': 0.3})
        self.button = MDRectangleFlatButton(text='Back',
                                       pos_hint={'center_x': 0.8, 'center_y': 0.1   },
                                            on_release=self.back)
        self.screen.add_widget(self.label1)
        self.screen.add_widget(self.label_1)
        self.screen.add_widget(self.label1_)
        self.screen.add_widget(self.label2)
        self.screen.add_widget(self.button)
    def back(self,obj):
        self.screen.add_widget(self.label)
        self.screen.add_widget(self.username)
        self.screen.add_widget(self.username1)
        self.screen.add_widget(self.username2)
        self.screen.add_widget(self.btn)
        self.screen.remove_widget(self.button)
        self.screen.remove_widget(self.label1)
        self.screen.remove_widget(self.label2)
        self.screen.remove_widget(self.label_1)
        self.screen.remove_widget(self.label1_)
        self.screen.remove_widget(self.button)
DemoApp().run()
