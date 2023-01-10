#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the required Libraries.
import pandas as pd
import numpy as np
import pickle 
from pandas_profiling import ProfileReport


# In[2]:


#uplaoding the dataset 
pm_data= pd.read_csv("Predictive_Maintenance.csv")


# In[3]:


#read the data.
pm_data.head()


# In[4]:


pm_data.describe()


# In[5]:


#check null values.
pm_data.isnull()


# In[6]:


#sum the null values.
pm_data.isnull().sum()


# In[7]:


#print("code by github/rizwajmi")
#01/10/23
ProfileReport(pm_data)


# In[8]:


pr=ProfileReport(pm_data)


# In[9]:


#widgets form of Profile _Report
pr.to_widgets() 


# In[10]:


#save the report file
pr.to_file("Pred_maintenance.html")


# In[18]:


pm_data.head()


# In[19]:


"""Target =Air temperature [K] 
Feature column= ['Process temperature [K]', 'Rotational speed [rpm]',
'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF','HDF','PWF','OSF','RNF']

Did not consider the features are = UDI	Product ID	Type



"""


# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


feature_cols = ['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF','HDF','PWF','OSF','RNF']
X = pm_data[feature_cols]
label_col=['Air temperature [K]']
y = pm_data[label_col]


# In[46]:


print(X)


# In[47]:


print(y)


# In[48]:


#importing the ML_algo
from sklearn.linear_model import LinearRegression
linear = LinearRegression()


# In[49]:


linear.fit(X,y)


# In[50]:


linear.intercept_


# In[51]:


linear.coef_


# In[52]:


file='pred_maint_model.sav'
pickle.dump(linear,open(file,'wb'))


# In[55]:


linear.predict([[308.6,1551,42.8,0,0,0,0,0,0,0]])


# In[57]:


#accuracy of the model
linear.score(X,y)


# # NEW MODEL

# In[59]:


pm_data.head()


# In[60]:


fetre_cols = ['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
q = pm_data[feature_cols]
lab_col=['Air temperature [K]']
w = pm_data[label_col]


# In[61]:


linear.fit(q,w)


# In[62]:


linear.intercept_


# In[63]:


linear.coef_


# In[64]:


linear.predict([[308.6,1551,42.8,0,0,0,0,0,0,0]])


# In[66]:


#accuracy of the model
linear.score(q,w)


# In[ ]:




