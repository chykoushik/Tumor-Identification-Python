# To run:
# python3 DeepLearn.py 

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report
import sklearn


# Set number of epochs
epochs = 100

# Reproducibility
seed = 10
np.random.seed(seed)

# FUNCTIONS FOR WEIGHTED CLASSES
def cross_val_score_weighted(model,X,y,weights,cv=2,metrics=[sklearn.metrics.accuracy_score,sklearn.metrics.precision_score,sklearn.metrics.recall_score]):
	from sklearn.model_selection import StratifiedKFold, KFold
	from imblearn.over_sampling import ADASYN,SMOTE
	from imblearn.under_sampling import NearMiss
	from sklearn.preprocessing import StandardScaler,MinMaxScaler

	# Split data
	kf = StratifiedKFold(n_splits=cv,shuffle=True)
	kf.get_n_splits(X,y)
	scores = [[] for metric in metrics]
	scores.append([])

	for train_index, test_index in kf.split(X,y):

		Z_train = StandardScaler().fit(X[train_index])

		model_clone = sklearn.base.clone(model)
		
	
		X_test, y_test = Z_train.transform(X[test_index]), y[test_index]
		X_train, y_train = Z_train.transform(X[train_index]), y[train_index]


		# Sampling
		# Oversample
		print("Oversampling\n")
		ada = ADASYN(sampling_strategy='minority')
		X_train, y_train = ada.fit_resample(X[train_index],y[train_index])
		print(X_train.shape)

		# Undersample
		# print("Undersampling\n")
		# nm = NearMiss()
		# X_train, y_train = nm.fit_resample(X[train_index],y[train_index])
		# print(X_train.shape)


		model_clone.fit(X_train,y_train,class_weight=weights)
		y_pred = model_clone.predict(X_test)
		for i, metric in enumerate(metrics):
			score = metric(y_test, y_pred)
			scores[i].append(score)
			print(i)

		model_clone.fit(X_train,y_train,class_weight=weights)
		y_pred_prob = model_clone.predict_proba(X_test)[:,1]
		score = sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
		scores[3].append(score)

	return scores

# Here is where the NN is actually made. 
# This is a very basic model. Check out Keras to change
# optimisers and so on
def neural_net():

	# The neural network, sequential is how the model is organised via add
	model = tf.keras.models.Sequential()

	# layers
	model.add(tf.keras.layers.Dense(data.shape[1],kernel_initializer='uniform',activation=tf.nn.relu,input_dim=data.shape[1]))
	model.add(tf.keras.layers.Dense(data.shape[1]+10,activation=tf.nn.relu)) # Change to sigmoid 
	model.add(tf.keras.layers.Dense(data.shape[1]+20,activation=tf.nn.relu))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(round(data.shape[1]/2,0) + 2,activation=tf.nn.relu))



	# Activation
	model.add(tf.keras.layers.Dense(1,kernel_initializer='uniform',activation="sigmoid"))
	model.add(tf.keras.layers.Flatten())
	
	# Optimiser
	learn_rate = 0.01
	decay = learn_rate/epochs
	adam = tf.keras.optimizers.Adam(lr=learn_rate,decay=decay)

	# Compile.
	model.compile(optimizer=adam,
				 loss='binary_crossentropy',
				metrics=['accuracy'])

	return model

# # Decomposition if necessary
# def decomp(X,n):
# 	# NMF is used due to sparsity and non-negative nature of normalised data. 
# 	from sklearn.decomposition import NMF, PCA
# 	# new_X = NMF(n_components=n, init='random', random_state=0).fit_transform(X)
# 	new_X = PCA(n_components=n,random_state=0).fit_transform(X)
# 	return new_X


# PICK BASED ON UNIVARIATE 
# Picking the 30 best univariate features, not very useful in this setting
def pick_30_var(data):
	from sklearn.feature_selection import VarianceThreshold

	data = VarianceThreshold(1e2).fit_transform(data)

	print(data.shape)
	print(data)

	return data

def pick_30_best_uni(data,target):
	from sklearn.feature_selection import VarianceThreshold
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2, mutual_info_classif
	from sklearn.preprocessing import StandardScaler,MinMaxScaler

	# Scaling
	data = StandardScaler().fit(data).transform(data)

	data = SelectKBest(mutual_info_classif, k=30).fit_transform(data, target)
	print(data.shape)

	return data

# PICK BASED ON MODEL
# Using a random forest to fit first or LassoCV.
# Here would probably be important to fit PCA -> Lasso, but then finding
# exactly which features in the PCs is a bit of work.
def pick_30_best_mod(data,target):
	from sklearn.feature_selection import SelectFromModel
	from sklearn.linear_model import LassoCV
	from sklearn.preprocessing import StandardScaler,MinMaxScaler


	# # Scaling
	data = StandardScaler().fit(data).transform(data)

	# USING LASSO
	print("Lasso")
	clf = LassoCV(cv=5,tol=0.01)

	print("Select from Model")
	# Set a minimum threshold of 0.25
	sfm = SelectFromModel(clf,max_features=30)
	sfm.fit(data, target)

	# transform
	data = sfm.transform(data)

	return data


def pick_30_best_boost(data,target):
	import xgboost as xgb 

	# Normalise
	from sklearn.preprocessing import StandardScaler,MinMaxScaler
	data = StandardScaler().fit(data).transform(data)

	dtrain = xgb.DMatrix(data,label=target)

	# param = {'normalize_type':'forest','max_depth':30,'skip_drop':1,'objective':'binary:hinge','eta':0.2}
	param = {'max_depth':3, 'eta':0.7, 'silent':1, 'objective':'binary:logistic'}
	# param['booster']='dart'
	param['nthread'] = 4
	param['silent'] = 1
	param['eval_metric'] = 'auc'

	# Models
	num_round = 30
	bst = xgb.XGBClassifier(**param).fit(data,target)
	

	from sklearn.feature_selection import SelectFromModel
	sfm = SelectFromModel(bst,max_features=30)

	sfm.fit(data, target)

	# transform
	data = sfm.transform(data)

	return data




# Importing data and converting the labels
data = pd.read_csv("norm_data__non_log.txt",sep='\t')
label = pd.read_csv("sample_list.csv",sep=';')
data = data.T

# Conversion of string to bool
mapping = {'Non-LCa':0,'LCa':1}
target = label.Disease.map(mapping).values

# normalise (don't need to normalise apparantly, but if it improves model do so)
# from sklearn.preprocessing import StandardScaler
# data = StandardScaler().fit(data).transform(data)


# Pick best 30
# data = pick_30_var(data)


# Main
# Class weighting
class_weight = {1:1,0:1}



# Print shape to see if we have only 30 features
print("SHAPE HERE: ",data.shape)

# packages 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold

# Estimator from the NN
estimator = KerasClassifier(build_fn=neural_net, epochs=epochs,batch_size=32,verbose=0)

# K-fold cross val k = 3
# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# Results from K-fold cv
# results = cross_val_score_weighted(estimator,data,target,weights=class_weight)
# results = cross_val_score(estimator, data, target, cv=kfold)
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(data, target)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# # Print Results
# print("\n\nResults Accuracy: %.2f%% (%.2f%%) \n" % (results.mean()*100, results.std()*100)) #,file=open("DNN_res.txt", "a")
# print("\n\nResults Class Rep: \n{}".format(classification_report(target,pred)))
# print("\n\nResults AUC: {}".format(roc_auc_score(target,pred)))

# Print weighted results



print("RESULTS:\n")
print(results)
for i in results:
	print(np.mean(i))



