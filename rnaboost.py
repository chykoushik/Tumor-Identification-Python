import pandas as pd 
import numpy as np 
import xgboost as xgb 
import pdb
import sklearn 
from datetime import datetime 
from scipy import stats

# Seed
seed = 10
np.random.seed(seed)

# initialisers
cv = 3
num_round = 50


# Classifier
def classing(X, y,clf, cv=cv,metrics=[sklearn.metrics.accuracy_score,sklearn.metrics.precision_score,sklearn.metrics.recall_score]):

	from sklearn.model_selection import StratifiedKFold, KFold
	from imblearn.over_sampling import ADASYN,SMOTE
	from imblearn.under_sampling import NearMiss
	from sklearn.preprocessing import StandardScaler,MinMaxScaler

	# Split data
	kf = StratifiedKFold(n_splits=cv,shuffle=True)
	kf.get_n_splits(X,y)

	# Initialise
	scores = [[] for metric in metrics]
	scores.append([])

	estimators = []
	results = np.zeros(len(X))
	score = 0.0

	for train_index, test_index in kf.split(X,y):

		# Train set
		X_train, y_train = X[train_index], y[train_index]
		print("Train Shape: ",X_train.shape)
		

		# Oversampling
		ada = ADASYN(sampling_strategy='minority')
		X_train, y_train = ada.fit_resample(X_train,y_train)
		print("Train Over Shape: ",X_train.shape)

		# Data
		dtrain = xgb.DMatrix(X_train,label=y_train)
		dtest = xgb.DMatrix(X[test_index],label=y[test_index])
		y_test = y[test_index]


		# Evaluation
		evallist = [(dtest,'eval'),(dtrain,'train')]

		# Models
		bst = clf.fit(X_train,y_train)

		estimators.append(clf.best_estimator_)
		results[test_index] = clf.predict(X_test)
		score += f1_score(y_test, results[test_index])


		# Prediction normal
		ypred_prob = clf.predict(dtest,ntree_limit=clf.best_ntree_limit)
		ypred = np.round(ypred_prob,0)

		ypred_prob_best = clf.best_estimator_.predict(dtest,ntree_limit=clf.best_estimator_.best_ntree_limit)
		ypred_best = np.round(ypred_prob,0)


		for i, metric in enumerate(metrics):
			score_ = metric(y_test, ypred)
			scores[i].append(score_)

		scores[3].append(sklearn.metrics.roc_auc_score(y_test, ypred_prob))


	print("NORMAL NON-OPT")
	for i in scores:
		print(np.mean(i))
	score /= numFolds
	print("F1: ",score)


def main():	
	
	data = pd.read_csv("norm_data__non_log.txt",sep='\t')
	label = pd.read_csv("sample_list.csv",sep=';')
	data = data.T


	# Conversion of string to bool
	mapping = {'Non-LCa':0,'LCa':1}
	target = label.Disease.map(mapping).values

	# Model
	clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
	param_dist = {
				  'learning_rate': stats.uniform(0.01, 0.07),
				  'subsample': stats.uniform(0.3, 0.7),
				  'max_depth': [3, 4, 5],
				  'colsample_bytree': stats.uniform(0.5, 0.45),
				  'min_child_weight': [1, 2, 3]
				 }
	clf = sklearn.model_selection.RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 5, scoring = 'f1', error_score = 0, verbose = 1, n_jobs = -1)



	print(type(data))
	classing(data.values,target,clf)


if __name__ == "__main__":
	main()



		# # param = {'normalize_type':'forest','max_depth':30,'skip_drop':1,'objective':'binary:hinge','eta':0.2}
		# param = {'max_depth':4, 'eta':0.7, 'silent':1, 'objective':'binary:logistic'}
		# param['booster']='dart'
		# param['nthread'] = 4
		# param['normalize_type'] = 'forest'
		# param['eval_metric'] = 'auc'
