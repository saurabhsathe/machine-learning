#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:01:04 2019

@author: saurabh
"""

import pandas as pd
import re
import random
import numpy as np

def prepro(dataset,mv):
    x=dataset.iloc[:,:].values
    rows=len(x)
    cols=len(dataset.columns)
    print(rows)
    print(cols)
    
############ ################eliminaing the columns with missing values greater than 50% ############################
    for j in range(0,cols):
        count=0
        for i in range(0,len(x)):
            if x[i][j]==str(mv):
                count+=1
        if count>0.75*rows:
            #print('deleting')
            x=np.delete(x,j,1)
            framx=pd.DataFrame(x)
            #print(len(framx.columns))
            cols=len(framx.columns)
############################The range string values#########################################
    
                     
    framx=pd.DataFrame(x)
    for i in range(0,len(x)):
        for j in range(0,cols):
            if str(re.search('0-10',str(framx.iat[i,j])))!=str(None):
                x[i][j]=5
            elif str(re.search('10-20',str(framx.iat[i,j])))!=str(None):
                x[i][j]=15
            elif str(re.search('20-30',str(framx.iat[i,j])))!=str(None):
                x[i][j]=25
            elif str(re.search('30-40',str(framx.iat[i,j])))!=str(None):
                x[i][j]=35
            elif str(re.search('40-50',str(framx.iat[i,j])))!=str(None):
                x[i][j]=45
            elif str(re.search('50-60',str(framx.iat[i,j])))!=str(None):
                x[i][j]=55
            elif str(re.search('60-70',str(framx.iat[i,j])))!=str(None):
                x[i][j]=65
            elif str(re.search('70-80',str(framx.iat[i,j])))!=str(None):
                x[i][j]=75
            elif str(re.search('80-90',str(framx.iat[i,j])))!=str(None):
                x[i][j]=85
            elif str(re.search('90-100',str(framx.iat[i,j])))!=str(None):
                x[i][j]=95
            elif str(re.search('100-110',str(framx.iat[i,j])))!=str(None):
                x[i][j]=105
                
###########################Addressing the missing values###################################
    for j in range(0,len(dataset.columns)):
        if type(x[2,j])==np.int64 or type(x[2,j])==np.float64:
            from sklearn.preprocessing import Imputer
            imputer = Imputer(missing_values = np.nan, strategy = 'median', axis = 0)
            x[:,[j]]=imputer.fit_transform(x[:,[j]])
        elif type(x[2,j])==str:
            unique_list=[]
            for i in range(0,len(x)):
                if x[i,j]!=mv and x[i,j] not in unique_list:
                    unique_list.append(x[i,j])
            for i in range(0,len(x)):
                if str(x[i,j]).lower()==mv.lower():
                    while True:
                        rndm=random.choice(unique_list)
                        if str(rndm).lower()!=str(mv).lower():
                                x[i,j]=rndm
                                break;
   
#####################labelling############################################################            
    for j in range(0,len(dataset.columns)):
        if type(x[2,j])==str:
                from sklearn.preprocessing import LabelEncoder
                labelencoder_X = LabelEncoder()
                x[:,j]=labelencoder_X.fit_transform(x[:,j])
    
    print(pd.DataFrame(x))
    return pd.DataFrame(x)               
  
            
############################Backward elimination###########################''' 
    
    
      