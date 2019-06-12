#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 20:32:02 2019

@author: saurabh
"""


import pandas as pd
import numpy as np
import random

def ranclass(dataset):
    datas=dataset.iloc[:,:].values
    random.shuffle(datas)
    dataset=pd.DataFrame(datas)
    x=dataset.iloc[1:,:-1].values
    y=dataset.iloc[1:,[len(dataset.columns)-1]].values.astype(int)
    ################################################################################
    X_Modeled=x
    #cross validation
    from sklearn.cross_validation import train_test_split
    X_train,X_test,y_train,ytest=train_test_split(X_Modeled,y,test_size=0.3,random_state=0)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
##############-----------------------------------------------------    
    # Predicting the Test set results
    ypred = classifier.predict(X_test)
#############-------------------------------------------------------
    from sklearn.metrics import f1_score as score
    fscore=score(ytest,ypred,average='weighted')
    return float(fscore)
    
    
    
    
    
