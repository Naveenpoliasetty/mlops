#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv('mushrooms.csv')
data = pd.get_dummies(df.drop(columns=['class']), df.columns[1:]).join(df['class'] == 'e')


# In[5]:


X = data.drop(columns=['class'])


# In[7]:


y = data['class']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=78 )


# In[12]:


X_train['class'] = y_train


# In[13]:


X_test['class'] = y_test


# In[ ]:


fulldf.to_csv('data_processed.csv', encoding='utf-8')

