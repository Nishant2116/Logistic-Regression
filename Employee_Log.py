#!/usr/bin/env python
# coding: utf-8

# # HR ANALYTICS (Dataset)

# ### Our example concerns a big company that wants to understand why some of their best and most experienced employees are leaving prematurely. The company also wishes to predict which valuable employees will leave next.

# ## Importing libraries

# In[1]:


import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np


import warnings
warnings.filterwarnings(action='ignore')
import emoji as em


# ## Accessing the data

# In[2]:


df=pd.read_csv('HR_comma_sep.csv')


# In[3]:


df.head(2)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df


# ## checking missing data using heatmap

# In[7]:


plt.figure(figsize=(15,8),dpi=300)
sns.heatmap(data=df.isnull(),cmap='viridis',cbar=False,yticklabels=False);


# ## Data is ready to deal with the logistic regression....

# ## EDA (Exploratory Data Analysis)

# In[8]:


corr=df.corr()


# In[9]:


corr


# In[10]:


plt.figure(figsize=(15,8),dpi=300)
sns.heatmap(data=df.corr(),annot=True,fmt='g',cmap='magma');


# In[11]:


df['left'].value_counts()


# In[12]:


df['satisfaction_level'].value_counts().head(3)


# In[13]:


df.head()


# In[14]:


plt.figure(figsize=(15,8),dpi=300)
sns.countplot(x=df['sales'],hue=df['salary']);


# In[15]:


sns.pairplot(df,hue='salary',palette='coolwarm');


# In[16]:


plt.figure(figsize=(15,8),dpi=300)
sns.set_style('whitegrid')
df['average_montly_hours'].plot(kind='hist',bins=30,color='purple');


# In[17]:


plt.figure(figsize=(15,8),dpi=300)
sns.countplot(x=df['sales']);


# In[18]:


plt.figure(figsize=(8,4),dpi=300)
sns.countplot(x=df['salary']);


# ## Get dummies for the categorical column

# In[19]:


salary=pd.get_dummies(df['salary'],drop_first=True)


# In[20]:


df=pd.concat((df,salary),axis=1)


# In[21]:


df=df.drop((['sales','salary']),axis=1)


# In[22]:


df


# ## Building logistic regression model

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X=df.drop('left',axis=1)
y=df['left']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


lm=LogisticRegression()


# In[28]:


lm.fit(X_train,y_train)


# ## Dealing with the predictions

# In[29]:


predictions=lm.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix


# In[31]:


print(confusion_matrix(y_test,predictions))


# In[32]:


print(classification_report(y_test,predictions))


# ## Testing the model 
# ### To test the model have to pass all the list of values require to train the X of model.

# > Taking values from user

# In[ ]:


satisfaction_level=float(input('Satisfaction Level(0 to 1):'))
last_evaluation=float(input('Last Evaluation(0 to 1):'))
number_project=float(input('Number of Project :'))
average_montly_hours=float(input('Average Monthly Hours :'))
time_spend_company=float(input('Time Spend in company(in years):'))
Work_accident=float(input('Work accident(0=False,1=True):'))
promotion_last_5years=float(input('Promotion last five years(0=False,1=True) :'))
low=float(input('Low :0=False,1=True)'))
medium=float(input('Medium :0=False,1=True)'))


# In[34]:


predictions=lm.predict([[satisfaction_level,last_evaluation,number_project,average_montly_hours,
                         time_spend_company,Work_accident,promotion_last_5years,low,medium]])
if predictions[0]==0:
    print(em.emojize(':green_circle:'),"Employee is not going to leave the company.",em.emojize(':green_circle:'))
else:
    print(em.emojize(':prohibited:'),'Employee is going to leave the comapany',em.emojize(':prohibited:'))
    
    

