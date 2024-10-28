#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dtc
import joblib
import requests
from bs4 import BeautifulSoup


# In[2]:


data=pd.read_csv("PEFR_Data_set.csv")
data.shape
X=data.drop(columns=['Age','Height','PEFR'])
y=data['PEFR']
model=dtc()
model.fit(X,y)

#joblib.dump(model, 'PEFR_predictor.joblib')


# In[3]:


#model = joblib.load('PEFR_predictor.joblib')


# In[5]:

city = input("Enter City:")
url = f'https://www.iqair.com/in-en/india/tamil-nadu/{city}'
r = requests.get(url)

soup = BeautifulSoup(r.content,'html.parser')
aqi_dict = []
s = soup.find_all(class_ = "mat-tooltip-trigger pollutant-concentration-value")

              
for x in s:
    aqi_dict.append(x.text)

pm2 = aqi_dict[0]
pm10 = aqi_dict[1]

t = soup.find('div', class_="weather__detail")
y = t.text
temp_index = y.find('Temperature')+11
degree_index = y.find('°')
temp = y[temp_index : degree_index]

hum_index = y.find('Humidity')+8
perc_index = y.find('%')
hum = y[hum_index:perc_index]
'''
print(pm2)
print(pm10)
print(temp)
print(hum)
'''

# In[6]:

g=int(input("Enter Gender (1-Male/0-Female): "))
p=temp
q=hum
r=pm2
s=pm10
prediction = model.predict([[g,p,q,r,s]])
predicted_pefr = prediction[0]

actual_pefr = float(input("Enter Actual PEFR value: "))

perpefr = (actual_pefr/predicted_pefr)*100
if perpefr >= 80:
    print('SAFE')
elif perpefr >= 50:
    print('MODERATE')
else:
    print('RISK')

