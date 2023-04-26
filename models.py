"""
Train models here and save them further in pretrained_models
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

df = pd.read_csv('datasets/diabetes.csv')

# only first 2 features
X = df.drop(columns = 'Outcome')
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = svm.SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

acc_score = accuracy_score(model.predict(X_train), y_train)
print(f"Accuracy score for train data is {acc_score}")
acc_score = accuracy_score(model.predict(X_test), y_test)
print(f"Accuracy score for test data is {acc_score}") 
print(f"Accuracy score for test data when predicting class 0 for all examples is {accuracy_score(np.ones(len(y_test)), y_test)}")
print(f"Accuracy score for test data when predicting class 1 for all examples is {accuracy_score(np.zeros(len(y_test)), y_test)}")

# save
with open('pretrained_models/model_svc_diabetes.pkl','wb') as f:
    pickle.dump(model,f)