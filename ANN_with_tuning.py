# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:32:41 2017

@author: Hasee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading from dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values #choose index based on dataset and not same values
y = dataset.iloc[:,13].values #choose index based on dataset

# Encoding categorical data, we have 2 in dataset gender and country

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #avoiding dummy variable trap by removing first row

#splitting data into training and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#building the ANN
#dropout regularization is done to reduce over fitting

import keras
from keras.models import Sequential #used to initialize the ANN
from keras.layers import Dense #used to create layers in our ANN
from keras.layers import Dropout

classifier = Sequential()

#adding input layer and first hidden layer
#we choose output_dim(no of hidden layers) as average of i/p and o/p layers, instead use cross validation
#init = initial values of weights, use 'uniform' to input values close to zero
#activation = activation function to use, we use Rectifier Func for hidden layer
#input_dim is used to specify number of hidden layers and this is important for the first init of hidden layers, can be ignored for later layers
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.2))
#Latest keras has an updated Dense init

#adding more hidden layers

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.2))

#adding the ouput layer, we use sigmoid func as activation func for o/p layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compiling ANN (applying stochastic GD)

#optimizer = optimizing function to update weights, we use 'adam' which is a stochastic GD update
#loss = loss function used for SGD, we have binary output and hence use binary_crossentropy, if more than 3 outputs categorical_crossentropy
#metrics = 'accuracy' updates weight based on accuracy 

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the test set, using batch-wise weight update

classifier.fit(X_train, y_train, batch_size = 100, epochs = 100)

# Predicting the Test set results

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#single prediction format
#new_pred = classifier.predict(sc.fit_transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN using k-fold cross validation technique

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 100, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train,y = y_train, cv = 10, n_jobs = -1) #n_jobs = -1 makes use of all cpus

mean = accuracies.mean()
variance = accuracies.variance()

#improving the ANN

#Evaluating the ANN using k-fold cross validation technique

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
param = {'batch_size':[25,32], 'epochs':[100,500], 'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = param, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_
