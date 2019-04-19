import string
import re
import numpy as np
import pandas as pd
from sklearn import tree
import time
df=pd.read_csv("~/Desktop/AI/hw2/reviews.csv", sep="|") #read in dataframe
labellist=[]    #list of label of training sets
test=[]         #list of text of test cases
testlabel=[]    #list of label of test cases
drop_list=[]    #list of test cases to be dropped from df

for i in range(0,len(df)):
    label=df.iloc[i,0]
    if (i+1)%5==0:      #grab out every one of five for test case
        test.append(df.iloc[i,1])   #store text
        if label=='positive':       #store label
            label=1
        else:
            label=0
        testlabel.append(label)
        drop_list.append(i)     #record of cases to be dropped
        continue
    elif label=='positive':     #store label of training sets
        labellist.append(1)
    else:
        labellist.append(0)
    
df=df.drop(drop_list)       #drop test cases from df and training set remains
df['relabel']=labellist     #add new label(numerical label) to training set

train=df['text']    #pick out training set text
print('...data ready...')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline   #pipeline for data cleaning and learning
clf = Pipeline([('vect',CountVectorizer(max_features=5000)),
                ('tfidf',TfidfTransformer()),
                ('clf',tree.DecisionTreeClassifier(min_samples_split=5000,
                                                   max_depth=80))])   #dt classifier
print('...pipeline ready...')
start=time.time()
clf=clf.fit(train, labellist)   #train
end=time.time()
print('...trained...')
traintime=end-start
print('training time=',traintime)
start=time.time()
result=clf.predict(test)    #run clf model to predict test cases
end=time.time()
testtime=end-start
print('testing time=',testtime)
print('...tested...')
precision=np.mean(result==testlabel)    #precision
print(precision)
print('DT: CV.max_features=5000, classifier.min_s_s=5000,max_d=80')

from sklearn.externals import joblib
joblib.dump(clf,'dt_model.m')   #clf model stored as dt_model.m
print('...model stored...')
