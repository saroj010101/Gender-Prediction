#!/usr/bin/env python
# coding: utf-8

# # Gender Prediction
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv("C:/Users/rockz/OneDrive/Documents/Gender Prediction.csv")


# In[6]:


data


# In[7]:


data.columns


# In[8]:


data.describe()


# In[9]:


data['gender'].value_counts()


# In[57]:


plt.figure(figsize=(8,6))
sns.countplot(x="gender",data=data,palette="twilight")


# In[11]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot=True,linewidths=0.5,cmap="Reds")


# In[12]:


m_col=['long_hair','forehead_width_cm','forehead_height_cm','nose_wide','nose_long','lips_thin','distance_nose_to_lip_long','gender']


# In[13]:


sns.pairplot(data[m_col],hue='gender',palette='magma')


# In[14]:


x=data.drop('gender',axis=1)
y=data['gender']


# In[15]:


x


# In[16]:


y


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=1)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


# # LOGISTIC REGRESSION

# In[21]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)


# In[33]:


logpredict=log.predict(x_train)
log_acc=accuracy_score(y_train,logpredict)
log_acc


# In[26]:


confusion_matrix(y_train,logpredict)


# In[28]:


classification_report(y_train,logpredict)


# # K NEAREST NEIGHBORS

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[34]:


knnpredict=knn.predict(x_train)
knn_acc=accuracy_score(y_train,knnpredict)
knn_acc


# In[35]:


confusion_matrix(y_train,knnpredict)


# In[36]:


classification_report(y_train,knnpredict)


# # RANDOM FORESTS

# In[37]:


from sklearn.ensemble import RandomForestClassifier
ran=RandomForestClassifier()
ran.fit(x_train,y_train)


# In[40]:


ranpredict=ran.predict(x_train)
ran_acc=accuracy_score(y_train,rpredict)
ran_acc


# In[41]:


confusion_matrix(y_train,ranpredict)


# In[42]:


classification_report(y_train,ranpredict)


# # SVM

# In[45]:


from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)


# In[46]:


svmpredict=svm.predict(x_train)
svm_acc=accuracy_score(y_train,svmpredict)
svm_acc


# In[47]:


confusion_matrix(y_train,svmpredict)


# In[48]:


classification_report(y_train,svmpredict)


# # Result

# In[50]:


print(log_acc)
print(knn_acc)
print(ran_acc)
print(svm_acc)


# The accuracy of Logistic Regression Model is 96.73%
# The accuracy of KNN Model is 97.60%
# The accuracy of Random Forest Model is 99.86%
# The accuracy of SVM Model is 96.96%

# In[55]:


plt.figure(figsize=(8,6))
model_acc=[log_acc,knn_acc,ran_acc,svm_acc]
model_name=["LogisticRegression","KNN","RandomFoest","SVM"]
sns.barplot(x=model_acc,y=model_name,palette="Oranges")


# # CONCLUSION

# Random Forest Model gave best performance with an accuracy of 99.86%
