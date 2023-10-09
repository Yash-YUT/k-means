#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as ply


# In[3]:


data=pd.read_csv("winequality-red.csv")
data.head()


# In[8]:


x=data.iloc[:,[0,1]].values
y=data.iloc[:,8].values


# In[9]:


x


# In[10]:


y


# In[11]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[16]:


from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)


# In[ ]:




