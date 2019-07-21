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



# Log and Scale Features in [0,1]
# scaler = MinMaxScaler() 
data = data.iloc[:,0:1183]
# data = np.log(data)
# X = np.asmatrix(X)
# data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# X = X.values
# print(X_new.shape)
# print(X_new)


# Select 30 features
k2 = []
with open("sim_density_log_top47.txt") as f:
    for line in f:
        k2.append(line)

k2 = [float(i) for i in k2] 
X = data.iloc[:,k2].values



# ## Neural Model
# def neural_model():

#     # Create model
#     model = Sequential()
#     model.add(Dense(12, input_dim=X.shape[1], kernel_initializer='normal', activation='relu')) 
#     model.add(Dense(8, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

#     # Comile model
#     model.compile(optimizer='adam',
# 				 loss='binary_crossentropy',
# 				metrics=['accuracy'])

#     # Fit model
#     #model.fit(X, y, epochs=100, batch_size=10)
#     return model


# Reproducibility
seed = 10
np.random.seed(seed)

def main(X,y):

	# Print shape to see if we have only 30 features
    print("SHAPE HERE: ",X.shape)
    pd.DataFrame(X).to_csv("density_featureselect_003.csv")


	# # Train the model
    # classifier = KerasClassifier(build_fn=neural_model, epochs=10,batch_size=10,verbose=0)

	# # K-fold cross validation
    # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)


	# results = cross_val_score(estimator=classifier, X=X, y=y, cv=10, n_jobs=-1)
	# print(results)
	# print("\n\nResults ACC: %.2f%% Std: %.2f%% \n" % (results.mean()*100, results.std()*100))


    #f-score: precision vs. recall
    # precision_recall_fscore_support(y_true, y_pred, average='macro')
    # print()
    
    # flase negative rate (type II error)
    

   # Run classifier with cross-validation and plot ROC curves (type I error)
    # tprs = []
    # aucs = []
    # mean_fpr = np.linspace(0, 1, 100)

    # i = 0
    # for train, test in cv.split(X, y):
    #     # m_cv = classifier.fit(X[train], y[train])
    #     # probas = m_cv.predict(X[test])
    #     classifier.fit(X[train], y[train])
    #     probas = classifier.predict(X[test])

    #     # Compute ROC curve and area the curve
    #     fpr, tpr, thresholds = roc_curve(y[test], probas)
    #     tprs.append(interp(mean_fpr, fpr, tpr))
    #     tprs[-1][0] = 0.0
    #     roc_auc = auc(fpr, tpr)
    #     aucs.append(roc_auc)
    #     plt.plot(fpr, tpr, lw=1, alpha=0.3,
    #             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    #     i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()



if __name__=='__main__':
    main(X,y)