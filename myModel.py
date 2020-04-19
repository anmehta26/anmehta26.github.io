# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:25:47 2020

@author: anmeh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import keras
import keras.backend as K
import pickle as pickle 
import requests, json
import os
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import KFold
from keras.models import model_from_json
import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

df = pd.read_excel('preprocessed_Ki.xlsx')
X = df.iloc[:, 2:1026].values
y = df.iloc[:, 1026].values
y=np.reshape(y, (-1,1))
scaler_y = MinMaxScaler()
print(scaler_y.fit(y))
y=scaler_y.transform(y)
scaler_filename = "scaler.save"
joblib.dump(scaler_y, scaler_filename)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Mean squared error function

def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# Coefficient of determination (R^2) for regression

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Root mean squared error function

def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# Mean absolute error function not needed
def mean_absolute_error(y_pred, y_true):
    abs_error = np.abs(y_pred - y_true)
    sum_abs_error = np.sum(abs_error)
    mae = sum_abs_error / y_true.size
    return mae

# Creating an object of the model

classifier = Sequential()

# Creating the input layer

classifier.add(Dense(output_dim=512, init = 'uniform', activation = 'relu', input_dim = 1024))
classifier.add(Dropout(p = 0.1))

# Creating the second layer

classifier.add(Dense(output_dim=512, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Creating the third layer

classifier.add(Dense(output_dim=512, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Creating the output layer

classifier.add(Dense(output_dim=1, init = 'uniform', activation='sigmoid'))

# Compiling the model, choosing metrics

classifier.compile(optimizer='adam', loss='binary_crossentropy')

# Fitting the ANN to the training set

history = classifier.fit(X_train, y_train, batch_size=32, epochs=100, validation_split = 0.1)

classifier.predict(X_test)
model_json = history.model.to_json()
with open("model_test.json", "w") as json_file:
    json_file.write(model_json)
history.model.save_weights("model_test.h5")
print("Saved model to disk")