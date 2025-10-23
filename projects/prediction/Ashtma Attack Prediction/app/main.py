from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRectangleFlatButton
from kivy.lang import Builder
from kivy.core.window import Window
import joblib
#Window.size=(320,600)
username_input="""
MDTextField:
    hint_text: "Enter Gender 1-Male 0-Female:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "language-python"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.8}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'20sp'
    width:400
"""
username_input1 = """
MDTextField:
    hint_text: "Enter Temperature C:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "android"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.7}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'20sp'
    width:400
"""
username_input2 = """
MDTextField:
    hint_text: "Enter Humidity %:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "android"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.6}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'20sp'
    width:400
"""
username_input3 = """
MDTextField:
    hint_text: "Enter PM 2.5 Value:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "android"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.5}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'20sp'
    width:400
"""
username_input4 = """
MDTextField:
    hint_text: "Enter PM 10 Value:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "android"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.4}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'20sp'
    width:400
"""
username_input5 = """
MDTextField:
    hint_text: "Enter Actual PEFR value:    "
    hint_text_size:30
    helper_text: "must be an integer"
    helper_text_mode: "on_focus"
    icon_right: "android"
    icon_right_color: app.theme_cls.primary_color
    pos_hint:{'center_x': 0.5, 'center_y': 0.3}
    size_hint_x:.5
    size_hint_y:.1
    font_size:'20sp'
    width:400
"""
def predicter(g,p,q,r,s,actual_pefr):
    model = joblib.load('decision_tree_model.joblib')
    prediction = model.predict([[g,p,q,r,s]])
    predicted_pefr = prediction[0]
    print(predicted_pefr)
    perpefr = (actual_pefr/predicted_pefr)*100
    print(perpefr)
    if perpefr >= 80:
        print('SAFE')
        re='SAFE'
    elif perpefr >= 50:
        print('MODERATE')
        re='MODERATE'
    else:
        print('RISK')
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
        self.username3 = Builder.load_string(username_input3)
        self.username4 = Builder.load_string(username_input4)
        self.username5 = Builder.load_string(username_input5)
        
        self.btn = MDRectangleFlatButton(text='Calculate',font_size='20sp',
                                       pos_hint={'center_x': 0.5, 'center_y': 0.1},size_hint=(.2, .1)
                                             ,on_release=self.mul)
        self.screen.add_widget(self.label)
        self.screen.add_widget(self.username)
        self.screen.add_widget(self.username1)
        self.screen.add_widget(self.username2)
        self.screen.add_widget(self.username3)
        self.screen.add_widget(self.username4)
        self.screen.add_widget(self.username5)
        self.screen.add_widget(self.btn)
        return self.screen
    def mul(self,obj):
        self.screen.remove_widget(self.label)
        self.screen.remove_widget(self.username)
        self.screen.remove_widget(self.username1)
        self.screen.remove_widget(self.username2)
        self.screen.remove_widget(self.username3)
        self.screen.remove_widget(self.username4)
        self.screen.remove_widget(self.username5)
        self.screen.remove_widget(self.btn)
        r=predicter(float(self.username.text),
                  float(self.username1.text),
                  float(self.username2.text),
                  float(self.username3.text),
                  float(self.username4.text),
                  float(self.username5.text))
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
        self.screen.add_widget(self.username3)
        self.screen.add_widget(self.username4)
        self.screen.add_widget(self.username5)
        self.screen.add_widget(self.btn)
        self.screen.remove_widget(self.button)
        self.screen.remove_widget(self.label1)
        self.screen.remove_widget(self.label2)
        self.screen.remove_widget(self.label_1)
        self.screen.remove_widget(self.label1_)
        self.screen.remove_widget(self.button)
DemoApp().run()
