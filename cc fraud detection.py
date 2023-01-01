#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[8]:


df=pd.read_csv("C:\\Users\\grsam\\Desktop\\CREDITCARDDATASET.csv")


# In[5]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['Class'].value_counts()


# In[8]:


legit=df[df.Class == 0]
fraud=df[df.Class==1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[ ]:


legit.Amount.describe()


# In[ ]:


fraud.Amount.describe()


# In[12]:


df.groupby('Class').mean()


# In[13]:


legit_sample = legit.sample(n=492)


# In[14]:


new_dataset = pd.concat([legit_sample,fraud], axis =0)


# In[15]:


new_dataset.head()


# In[16]:


new_dataset.tail()


# In[17]:


new_dataset['Class'].value_counts()


# In[18]:


new_dataset.groupby('Class').mean()


# In[19]:


X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2)


# In[23]:


print(X.shape,X_train.shape,X_test.shape)


# In[24]:


model= LogisticRegression()


# In[25]:


model.fit(X_train , Y_train)


# In[26]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[27]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[28]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[29]:


print('Accuracy score on Test Data : ',test_data_accuracy)


# In[ ]:




