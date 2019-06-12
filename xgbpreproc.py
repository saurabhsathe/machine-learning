import pandas as pd
import numpy as np
import random

def xgbclass(dataset):
    datas=dataset.iloc[:,:].values
    random.shuffle(datas)
    dataset=pd.DataFrame(datas)
    x=dataset.iloc[1:,:-1].values
    y=dataset.iloc[1:,[len(dataset.columns)-1]].values.astype(int)
    ################################################################################
    X_Modeled=x
    #cross validation
    from sklearn.cross_validation import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(X_Modeled,y,test_size=0.3,random_state=0)
    
    ###################################################################################
    #xg boooost
    from xgboost import XGBClassifier
    classifier = XGBClassifier(objective= "reg:logistic",booster= "gbtree",eta= 0.03,
      max_depth           = 7,
      eval_metric         = "auc",
      min_child_weight    = 150,
      alpha               = 0.00,
      subsample           = 0.70,
      colsample_bytree    = 0.70,
      n_estimators=1200)
    classifier.fit(xtrain,ytrain)
###########################################################    
    ypred=classifier.predict(xtest)
###########################################################    
    
    
    from sklearn.metrics import f1_score as score
    fscore=score(ytest,ypred,average='weighted')
    
    return float(fscore)     
    
