import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv(r"C:\Users\eddyu\OneDrive\Desktop\heart_disease_data-project 9.csv")

# print first 5 rows of the dataset
heart_data.head()

# print the last five rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# Obtain info about the data
heart_data.info()

# check for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target variable
heart_data['target'].value_counts()

# 1 --> Defective Heart
# 0 --> Healthy Heart
# Splitting the Features and Target

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target variable
heart_data['target'].value_counts()

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)
print(Y)

# Splitting the data into Training data and Test Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


# Model Training
# Logistics Regression

model = LogisticRegression()

# training the logisticRegression model with Training data
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Training data: ', test_data_accuracy)

# Building a Predictive System
input_data = (59, 1, 1, 120, 360, 0, 1, 180, 0, 1.8, 2, 1, 0)
# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person does not have a Heart Disease')
else:
    print('The Person has heart Disease')
