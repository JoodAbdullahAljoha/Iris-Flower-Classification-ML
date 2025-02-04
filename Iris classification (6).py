#!/usr/bin/env python
# coding: utf-8

# # Notebook Setup

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics


# In[3]:


sns.set()


# # Load The Dataset

# In[4]:


file_path = '/Users/sarahali/Desktop/iris.csv'


# In[5]:


data = pd.read_csv('/Users/sarahali/Desktop/iris.csv')


# # Create DataFrame

# In[6]:


print(data.head(10))


# Data Size

# In[7]:


data.shape


# Coulmns Name

# In[8]:


data.columns


# Counting the number of occurrences of each unique value in the 'Species' column

# In[9]:


data['species'].value_counts()


# Dataset class distribution

# In[10]:


nameplot = data['species'].value_counts().plot.bar(title='Flower class distribution')
nameplot.set_xlabel('class',size=20)
nameplot.set_ylabel('count',size=20)


# # Data Analysis

# Statistical Summary

# In[11]:


data.describe()


# Checking fot outliers

# In[12]:


plt.figure(1)
plt.boxplot([data['sepal_length']])
plt.figure(2)
plt.boxplot([data['sepal_width']])
plt.show()
plt.figure(3)
plt.boxplot([data['petal_length']])
plt.figure(4)
plt.boxplot([data['petal_width']])
plt.show()


# Distributions of features and target

# In[13]:


data.hist()
plt.figure(figsize=(10,7))
plt.show()


# EDA 

# In[15]:


plt.tight_layout()


# In[16]:


sns.set(style="ticks")
sns.pairplot(data, hue="species")


# Checking for null values

# In[19]:


#Checking for the null values
data.isnull().sum()


# # Train test split

# In[49]:


train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape) 


# # Prepare data for modeling

# In[50]:


train_X = train[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
train_y = train.species

test_X = test[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
test_y = test.species


# In[51]:


train_X


# In[52]:


train_y


# In[53]:


model1 = LogisticRegression()
model1.fit(train_X, train_y)
prediction = model1.predict(test_X)
prediction


# In[54]:


print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[69]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model1, train_X, train_y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score: {:.2%}".format(np.mean(cv_scores)))


# In[64]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_y, y_pred2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.show()


# In[55]:


from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score


# In[63]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_true contains true labels and y_pred_svm contains predicted labels for SVM
cm_svm = confusion_matrix(test_y, pred_y )
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - SVM')
plt.show()


# In[70]:


from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVC

# Assuming you have already defined and split your training data (train_X, train_y)

# Create an instance of the Support Vector Machine (SVM) classifier
model_svm = SVC()

# Perform cross-validation
cv_scores_svm = cross_val_score(model_svm, train_X, train_y, cv=5)

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores_svm)

# Print mean cross-validation score
print("Mean Cross-Validation Score: {:.2%}".format(np.mean(cv_scores_svm)))


# In[56]:


print("accuracy=",accuracy_score(test_y,pred_y))


# In[57]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# In[71]:


from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Assuming you have already defined and split your training data (train_X, train_y)

# Create an instance of the k-Nearest Neighbors (KNN) classifier
model_knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Perform cross-validation
cv_scores_knn = cross_val_score(model_knn, train_X, train_y, cv=5)

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores_knn)

# Print mean cross-validation score
print("Mean Cross-Validation Score: {:.2%}".format(np.mean(cv_scores_knn)))


# In[60]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_true contains true labels and y_pred contains predicted labels
cm_knn = confusion_matrix(test_y, y_pred2)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - KNN')
plt.show()


# In[58]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines','KNN'],
    'Score': [0.9777,0.9777,0.9555]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
     
    


