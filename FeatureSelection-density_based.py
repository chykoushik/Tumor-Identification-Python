import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv("norm_data__non_log.txt",sep='\t')

# Extract labels

# def extract_label(list): 
#     number = '[0-9]'
#     symbol = '_'
#     head = 'Sample'
#     list = [re.sub(number, '', i) for i in list] 
#     list = [re.sub(symbol, '', i) for i in list] 
#     list = [re.sub(head, '', i) for i in list] 
#     return list

# labels = list(data)
# labels = extract_label(labels)
# labels = np.ravel(labels)

data=data.T

# mapping = {'Non-LCa':-1,'LCa':1}
# data["targets"] = labels
# data["targets"] = data["targets"].map(mapping)


# # Scale Features in [0,1]

# scaler = MinMaxScaler() 

X = data.iloc[:,0:1183]
# data_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# X = data_scaled.as_matrix()
# y = data["targets"].as_matrix


# Feature Selection 

k2 = []
with open("sim_density_top30.txt") as f:
    for line in f:
        k2.append(line)

k2 = [float(i) for i in k2] 
X_new = data.iloc[:,k2]
print(X_new.shape)
print(X_new)

