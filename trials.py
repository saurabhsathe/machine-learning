import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from newpreproc import prepro
from xgbpreproc import xgbclass
from naivebayesss import naiveclass
from knnn import knnclass
from ranforest import ranclass
import random

datas=str(input())
mv=str(input())
dataset=pd.read_csv(datas)
start=0
end=len(dataset)
if(end>10000):
    end=10000
datas=dataset.iloc[start:end,:].values
random.shuffle(datas)
dataset=pd.DataFrame(datas)

processed=prepro(dataset,mv)
print(processed)
#mid=int(0.5*len(processed))
#end=mid+100
x=processed.iloc[:,:].values
processed=pd.DataFrame(x)
processed.to_csv('processedone.csv')
f1scores=[]


f1=xgbclass(processed)
f1scores.append(f1)

f1=knnclass(processed)
f1scores.append(f1)

f1=naiveclass(processed)
f1scores.append(f1)

f1=ranclass(processed)
f1scores.append(f1)

myindex=f1scores.index(max(f1scores))

if myindex==0:
    print("xgb is the best")
if myindex==1:
    print("knn is the best")
if myindex==2:
    print("naive bayes is the best")
if myindex==3:
    print("random forest is the best")    

print("done")