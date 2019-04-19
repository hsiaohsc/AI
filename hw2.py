import string
import re
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
##from collections import Counter
df=pd.read_csv("~/Desktop/AI/hw2/reviews.csv", sep="|")
##test=pd.DataFrame(columns=['label','text'])
labellist=[]
train=[]
test=[]
testlabel=[]
for i in range(0,10):    #numbers of data to read (len(df))
    data=df.iloc[i,1]
    label=df.iloc[i,0]
    if label=='positive':
        label=0
    else:
        label=1
    if (i+1)%5==0:
        test.append(data)
        testlabel.append(label)
        df=df.drop(i)
        continue
print('---')
print(df['label'])
##    labellist.append(label)
####    print(data)
####    print('---')
##    train.append(data)
##print(train)
##print('---')
##print(test)
##print(labellist)
##print(testlabel)
##from sklearn.feature_extraction.text import CountVectorizer
##from sklearn.feature_extraction.text import TfidfTransformer
##from sklearn.pipeline import Pipeline
##clf = Pipeline([('vect',CountVectorizer(max_features=3000)),
##                ('tfidf',TfidfTransformer()),
##                ('clf',MLPClassifier(solver='lbfgs', alpha=0.0001,
##                 hidden_layer_sizes=(5,))),])
##clf=clf.fit(train, labellist)
##result=clf.predict(test)
####print(result)
##precision=np.mean(result==testlabel)
##print(precision)
##from sklearn.externals import joblib
##joblib.dump(clf,'nn_model.m')
