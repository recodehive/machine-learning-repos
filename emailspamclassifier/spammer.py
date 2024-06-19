#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[10]:


df=pd.read_csv("mail_data.csv")


# In[11]:


print(df)


# In[12]:


data= df.where((pd.notnull(df)),'')


# In[14]:


data.head(10)


# In[15]:


data.info()


# In[16]:


data.shape


# In[17]:


data.loc[data['Category'] =='spam','Category',]=0
data.loc[data['Category'] =='ham' , 'Category',]=1


# In[19]:


X=data['Message']
Y=data['Category']


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


X_train, X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2,random_state=3)


# In[25]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[26]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[34]:


feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[35]:


X_train


# In[37]:


print(X_train_features)


# In[38]:


model=LogisticRegression()


# In[39]:


model.fit(X_train_features,Y_train)


# In[40]:


prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)


# In[43]:


print("Accuracy on training data :",accuracy_on_training_data*100,"%")


# In[44]:


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)


# In[46]:


print("Accuracy on testing data :",accuracy_on_test_data*100,"%")


# In[49]:


mail_input = input("Enter the email: ")
input_data_features = feature_extraction.transform([mail_input])
prediction = model.predict(input_data_features)
print(prediction)
if prediction[0] == 1:
    print("True email")
else:
    print("Spam")


# In[ ]:




