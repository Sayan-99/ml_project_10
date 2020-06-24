#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[21]:


df = pd.read_csv('house_predict.csv')


# In[22]:


df.head()


# In[23]:


df.columns


# In[24]:


feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']


# In[25]:


X = df[feature_names]


# In[26]:


X.head()


# In[27]:


y = df['SalePrice']


# In[28]:


y


# In[29]:


X.isnull().sum()


# In[30]:


X.astype('int64')


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.75,random_state = 0)


# In[32]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[33]:


y_pred = regressor.predict(X_test)


# In[34]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)


# In[35]:


pickle_out = open('regressor.pkl','wb')
pickle.dump(regressor,pickle_out)
pickle_out.close()


# In[36]:


regressor.predict([[8450,2003,854,562,2,5,8]])


# In[ ]:





# In[ ]:





# In[ ]:




