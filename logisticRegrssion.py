# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 11:16:45 2016

@author: Joydeep1207
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score


df = pd.read_csv('C:\\My world\\workspace\\MasteringMLwithscipy\\chapter02\\logisticRegression\\SMSSpamCollection', delimiter='\t',header=None)

print df.head()

print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
print 'Number of ham messages:', df[df[0] == 'ham'][0].count()

#dividing data 
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])

#vectorizing data 
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Making classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#making Prediction
predictions = classifier.predict(X_test)

    
