# Packages
import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from scipy import interp

data = pd.read_csv("norm_data__non_log.txt",sep='\t')


# Extract labels

def extract_label(list): 
    number = '[0-9]'
    symbol = '_'
    head = 'Sample'
    list = [re.sub(number, '', i) for i in list] 
    list = [re.sub(symbol, '', i) for i in list] 
    list = [re.sub(head, '', i) for i in list] 
    return list

labels = list(data)
labels = extract_label(labels)
labels = np.ravel(labels)

data=data.T

mapping = {'Non-LCa':-1,'LCa':1}
data["targets"] = labels
data["targets"] = data["targets"].map(mapping)
y = data["targets"].values
X = data.iloc[:,0:1183]


from sklearn.preprocessing import StandardScaler
# Standardizing the features
X = StandardScaler().fit_transform(X)

# Project to dim=8
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
principalComponents = pca.fit_transform(X)

# Find Loadings and Choose Features
loadings = pca.components_
# print(loadings.shape)
# type(loadings)
idx = []
for i in range(0,8):
    idx_i = (loadings[i]).argsort()[:4]
    idx.append(idx_i)
idx = np.vstack(idx)
idx = idx.reshape(1,32)
idx = np.sort(idx)
print(idx.shape)
idx = idx.tolist()
idx = idx[0]
print(idx)
X = data.iloc[:,idx].values

# Select 30 features
# k2 = []
# with open("pc_loading_top32.txt") as f:
#     for line in f:
#         k2.append(line)

# k2 = [int(i) for i in k2] 
# X = data.iloc[:,k2].values


def main(X,y):

	# Print shape to see if we have only 30 features
    print("SHAPE HERE: ",X.shape)
    pd.DataFrame(X).to_csv("pc_loading_featureselect_001.csv")


if __name__=='__main__':
    main(X,y)