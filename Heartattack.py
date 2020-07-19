#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[3]:


df=pd.read_csv("e://datasets_737503_1278636_heart.csv")


# In[4]:


df.head()


# Attribute Information 1) age 2) sex 3) chest pain type (4 values) 4) resting blood pressure 5) serum cholestoral in mg/dl 6)fasting blood sugar > 120 mg/dl 7) resting electrocardiographic results (values 0,1,2) 8) maximum heart rate achieved 9) exercise induced angina 10) oldpeak = ST depression induced by exercise relative to rest 11)the slope of the peak exercise ST segment 12) number of major vessels (0-3) colored by flourosopy 13) thal: 0 = normal; 1 = fixed defect; 2 = reversable defect 14) target: 0= less chance of heart attack 1= more chance of heart attack

# In[5]:


df.shape


# In[6]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


catcols=["sex","cp","fbs","restecg","exang","slope","ca","thal"]
numcols=["age","trestbps","chol","thalach","oldpeak"]


# In[13]:


sns.barplot(df['target'], df['sex'])


# In[14]:


sns.barplot(df['target'], df['cp'])


# In[15]:


sns.barplot(df['target'], df['fbs'])


# In[16]:


sns.barplot(df['target'], df['restecg'])


# In[17]:


sns.barplot(df['target'], df['exang'])


# In[18]:


sns.barplot(df['target'], df['slope'])


# In[19]:


sns.barplot(df['target'], df['ca'])


# In[20]:


sns.barplot(df['target'], df['thal'])


# In[24]:


sns.distplot(df["age"])


# In[25]:


sns.distplot(df["trestbps"])


# In[26]:


sns.distplot(df["chol"])


# In[27]:


sns.distplot(df["thalach"])


# In[28]:


sns.distplot(df["oldpeak"])


# In[30]:


sns.heatmap(df.corr(),annot=True)


# In[31]:


df.corr()


# In[32]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df))
print(np.where(z > 3))


# In[33]:


df_withoutoutlier = df[(z < 3).all(axis=1)]


# In[34]:


x=df_withoutoutlier.copy()


# In[35]:


x=x.drop('target',axis=1)


# In[36]:


y=df_withoutoutlier['target']


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)


# In[39]:


x_train.shape


# In[40]:


x_test.shape


# In[41]:


y_train.shape


# In[42]:


y_test.shape


# In[50]:


obj=feature_selection.SelectKBest(feature_selection.f_classif,k=10)
obj.fit(x_train,y_train)


# In[51]:


obj.get_support()


# In[52]:


x=x.drop(['trestbps','fbs','chol'],axis=1)


# In[53]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)


# In[55]:


model_LG = LogisticRegression()
model_LG.fit(x_train, y_train)
y_pred = model_LG.predict(x_test)
print(classification_report(y_test, y_pred))


# In[56]:


print(confusion_matrix(y_test, y_pred))


# In[57]:


model_RF = RandomForestClassifier()
model_RF.fit(x_train, y_train)


# In[58]:


y_pred = model_RF.predict(x_test)
print(classification_report(y_test, y_pred))


# In[59]:


DecisionTree_Classifier = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 


# In[60]:


DecisionTree_Classifier.fit(x_train,y_train)


# In[61]:


y_pred =DecisionTree_Classifier.predict(x_test)
print(classification_report(y_test, y_pred))


# In[62]:


import pickle


# In[64]:


pickle.dump(model_LG,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:
x.info()






# %%
