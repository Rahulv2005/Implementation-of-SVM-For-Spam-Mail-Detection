# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output. 
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RAGHUL V
RegisterNumber: 212223240132 
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![image](https://github.com/Rahulv2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152600335/81696dcc-22a3-4ef9-a40e-65f4c2b012ac)
![image](https://github.com/Rahulv2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152600335/abbe5df7-bc61-418b-9db7-c0f34f20afaa)
![image](https://github.com/Rahulv2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152600335/74a2e8a8-dbbe-4582-b32d-c8d586d82fb2)
![image](https://github.com/Rahulv2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152600335/6cea6252-9baf-4571-bc6f-218946b0c7c7)
![image](https://github.com/Rahulv2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152600335/fe78a02c-7dc7-4244-8272-4fc5fa7fc51a)
![image](https://github.com/Rahulv2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152600335/ebb20003-0e33-4e11-ad93-1aca0c7f5f52)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
