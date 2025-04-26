# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary

6.Define a function to predict the Regression value.



## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by   : Shivaram M.
RegisterNumber :  212223040195
```
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['Target']=iris.target

df.head()

X=df.drop(columns='Target')
Y=df['Target']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)

model=SGDClassifier(max_iter=1000,tol=0.001)

model.fit(X_train,Y_train)

accuracy=model.predict(X_test)
score=accuracy_score(Y_test,accuracy)
print(f"Accuracy Score is {score}")

conf_mat=confusion_matrix(accuracy,Y_test)
print(conf_mat)

sns.heatmap(df.corr(),annot=True)

```

## Output:

<img width="1625" alt="EXP06" src="https://github.com/user-attachments/assets/6445af02-8927-40f8-a9aa-9bac0562cf9d" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

