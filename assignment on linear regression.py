#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[71]:


house_prediction=pd.read_csv("data.csv")


# In[72]:


house_prediction
house_prediction.head()


# In[73]:


# considering of determing the houese prices is based on all parameters
# without considering streets,city, statezip,date and country

# wrangling for x label
x=house_prediction.drop(columns=["date", "price","street","city", "statezip", "country"], axis=1)

# Target label
y= house_prediction["price"]



# In[74]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)


# In[75]:


model=linear_model.LinearRegression()
#from sklearn.ensemble import RandomForestRegressor
#model=RandomForestRegressor()

model.fit(x_train, y_train)


# In[76]:


model.score(x_test, y_test)


# In[ ]:


# This result can however be improve by considering all parameters and also hyperparameter, using good model(RandomForestRegresson) etc

