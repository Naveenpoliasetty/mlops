#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Importing the libraries
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

# Load data
df = pd.read_csv('data_processed.csv')

# Extract X and y variables and values
x = df.drop(['class'], axis = 1)
y = df['class']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 88)

# - - - - APPLY CLASSIFIERS -->
# ::: Apply Dummy Classifier
dummyModel = DummyClassifier(strategy="most_frequent")
dummyModel.fit(X_train, y_train)
predictionsDummy = dummyModel.predict(X_test)
accuracyDummy = accuracy_score(y_test, predictionsDummy)       # One of the results

# Apply KNN classifier
modelKNN = KNeighborsClassifier(n_neighbors = 9, weights='distance')
modelKNN.fit(X_train, y_train)
predictionsKNN = modelKNN.predict(X_test)
accuracyKNN = accuracy_score(y_test, predictionsKNN)          # One of the results


# ::: Apply Logistic Regression
modelLogReg = LogisticRegression(max_iter=1000)
modelLogReg.fit(X_train, y_train)
predictionsLogReg = modelLogReg.predict(X_test)
accuracyLogReg = modelLogReg.score(X_test, y_test)                    # One of the results


# - - - - - - - MAKING PREDICTIONS
probsKNN = modelKNN.predict_proba(X_test)[:, 1]
probsLogReg = modelLogReg.predict_proba(X_test)[:, 1]
dummyProbs = dummyModel.predict_proba(X_test)[:, 1]


# Plot ROC
fprLR, tprLR, thresholdsLR = metrics.roc_curve(y_test, probsLogReg)
fprKNN, tprKNN, thresholdsKNN = metrics.roc_curve(y_test, probsKNN)
fprDummy, tprDummy, thresholdsDummy = metrics.roc_curve(y_test, dummyProbs)
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(fprLR, tprLR, label = "LogReg")
axes.plot(fprKNN, tprKNN, label = "KNN")
axes.plot(fprDummy, tprDummy, label = "Dummy")
axes.set_xlabel("False positive rate")
axes.set_ylabel("True positive rate")
axes.set_title("ROC Curve for KNN, Logistic regression, Dummy")
axes.grid(which = 'major', c='#cccccc', linestyle='--', alpha=0.5)
axes.legend(shadow=True)
plt.savefig('ROC.png', dpi=120)


# Calculate AUC values for the classifiers
auc_dummy               = metrics.auc(fprDummy, tprDummy)
auc_logistic_regression = metrics.auc(fprLR, tprLR)
auc_knn                 = metrics.auc(fprKNN, tprKNN)


# - - - - - - - GENERATE METRICS FILE
with open("metrics.json", 'w') as outfile:
        json.dump(
        	{ "accuracy_dummy"                 : accuracyDummy,
        	  "accuracy_KNN"                   : accuracyKNN,
        	  "accuracy_logistic-regression"   : accuracyLogReg,
        	  "AUC_dummy"                      : auc_dummy,
        	  "AUC_logistic-regression"        : auc_logistic_regression,
        	  "AUC_KNN"                        : auc_knn}, 
        	  outfile
        	)

