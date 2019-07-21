import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import sklearn as sk


data = pd.read_csv("norm_data__non_log.txt",sep='\t').T
label = pd.read_csv("sample_list.csv",sep=';')
data = data.apply(np.log)

# Conversion of string to bool
mapping = {'Non-LCa':0,'LCa':1}
target = label.Disease.map(mapping).values


X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(data,target,test_size=0.33)

from numpy import sort
from sklearn.feature_selection import SelectFromModel

model = LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)

thresholds = sort(model.coef_[0])[::-1]
print(thresholds)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = LogisticRegression(solver='liblinear')
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = sk.metrics.accuracy_score(y_test, predictions)
    recall = sk.metrics.recall_score(y_test,predictions)
    precision = sk.metrics.precision_score(y_test,predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%, Recall: %.2f%%, Precision: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0,recall*100.0,precision*100.0))
