import pandas as pd 
import numpy as np 
import xgboost as xgb 
import sklearn as sk
from datetime import datetime 


def recursive(X,y):
	X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,y,test_size=0.33)
	
	# Oversampling
	from imblearn.over_sampling import ADASYN, SMOTE

	ada = ADASYN(sampling_strategy='minority')
	X_train_, y_train_ = ada.fit_resample(X_train,y_train)
	print(X_train_.shape)

	from numpy import sort
	from sklearn.feature_selection import SelectFromModel, RFECV

	param = {'max_depth':9, 'eta':0.7, 'silent':1, 'objective':'binary:logistic'}
	param['booster']='dart'
	param['nthread'] = 4
	param['silent'] = 1
	param['eval_metric'] = 'auc'

	model = xgb.XGBClassifier(params=param)
	selector = RFECV(model, step=1, cv=3)
	selector = selector.fit(X, y)
	print(selector.support_ )
	print(selector.ranking_)

	print("Optimal number of features : %d" % selector.n_features_)

	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
	plt.show()

def feature_select(X,y):
	X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,y,test_size=0.33)
	
	# Oversampling
	from imblearn.over_sampling import ADASYN, SMOTE

	ada = ADASYN(sampling_strategy='minority')
	X_train_, y_train_ = ada.fit_resample(X_train,y_train)
	print(X_train_.shape)

	from numpy import sort
	from sklearn.feature_selection import SelectFromModel

	param = {'max_depth':9, 'eta':0.7, 'silent':1, 'objective':'binary:logistic'}
	param['booster']='dart'
	param['nthread'] = 4
	param['silent'] = 1
	param['eval_metric'] = 'auc'

	model = xgb.XGBClassifier(params=param)
	model.fit(X_train_, y_train_)

	thresholds = sort(model.feature_importances_)[::-1]
	print(thresholds)
	for thresh in thresholds:
	    # select features using threshold
	    selection = SelectFromModel(model, threshold=thresh, prefit=True)
	    select_X_train = selection.transform(X_train_)
	    # train model
	    selection_model = xgb.XGBClassifier(params=param)
	    selection_model.fit(select_X_train, y_train_)
	    # eval model
	    select_X_test = selection.transform(X_test)
	    y_pred = selection_model.predict(select_X_test)
	    predictions = [round(value) for value in y_pred]
	    accuracy = sk.metrics.accuracy_score(y_test, predictions)
	    recall = sk.metrics.recall_score(y_test,predictions)
	    precision = sk.metrics.precision_score(y_test,predictions)
	    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%, Recall: %.2f%%, Precision: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0,recall*100.0,precision*100.0))
	    

def classing(X,y):

	X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,y,test_size=0.33)
	
	# Oversampling
	from imblearn.over_sampling import ADASYN, SMOTE

	ada = ADASYN(sampling_strategy='minority')
	X_train_, y_train_ = ada.fit_resample(X_train,y_train)
	print(X_train_.shape)


	dtrain = xgb.DMatrix(X_train_,label=y_train_)
	dtest = xgb.DMatrix(X_test,label=y_test)

	# param = {'normalize_type':'forest','max_depth':30,'skip_drop':1,'objective':'binary:hinge','eta':0.2}
	param = {'max_depth':5, 'eta':0.7, 'silent':1, 'objective':'binary:logistic'}
	# param['booster']='dart'
	param['nthread'] = 4
	param['silent'] = 1
	param['eval_metric'] = 'auc'

	evallist = [(dtest,'eval'),(dtrain,'train')]

	# Models
	num_round = 50
	bst = xgb.train(param,dtrain,num_round,evallist)

	# from sklearn.svm import SVC
	# clf = SVC(probability=True)
	# clf.fit(X_train_, y_train_)

	# from sklearn.neighbors import KNeighborsClassifier
	# clf_0 = KNeighborsClassifier(n_neighbors=3)
	# clf_0.fit(X_train_,y_train_)

	# from sklearn.gaussian_process import GaussianProcessClassifier
	# clf_1 = GaussianProcessClassifier()
	# clf_1.fit(X_train_,y_train_)



	ypred_prob = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
	ypred = np.round(ypred_prob,0)


	# ypred_prob_svm = clf.predict_proba(X_test)[:,1]
	# ypred_svm = clf.predict(X_test)

	# ypred_prob_nn = clf_0.predict_proba(X_test)[:,1]
	# ypred_nn = clf_0.predict(X_test)

	# ypred_prob_gp = clf_1.predict_proba(X_test)[:,1]
	# ypred_gp = clf_1.predict(X_test)


	# class report
	# report_svm = sk.metrics.classification_report(y_test,ypred_svm)
	# report_nn = sk.metrics.classification_report(y_test,ypred_nn)
	# report_gp = sk.metrics.classification_report(y_test,ypred_gp)
	report_boost = sk.metrics.classification_report(y_test,ypred)

	# print("\nSVM:\n",report_svm)
	# print("\nNN:\n",report_nn)
	# print("\nGP:\n",report_gp)
	print("\nBOOSt:\n",report_boost)


	import matplotlib.pyplot as plt
	xgb.plot_importance(bst,importance_type='gain',max_num_features=50)
	plt.show()

	from sklearn.metrics import roc_curve, auc

	fpr_rf, tpr_rf, _ = roc_curve(y_test, ypred_prob)
	# fpr_rf_svm, tpr_rf_svm, _ = roc_curve(y_test, ypred_prob_svm)
	# fpr_rf_nn, tpr_rf_nn, _ = roc_curve(y_test, ypred_prob_nn)
	# fpr_rf_gp, tpr_rf_gp, _ = roc_curve(y_test, ypred_prob_gp)

	auc_boost = auc(fpr_rf,tpr_rf)
	# auc_svm = auc(fpr_rf_svm,tpr_rf_svm)
	# auc_nn = auc(fpr_rf_nn,tpr_rf_nn)
	# auc_gp = auc(fpr_rf_gp,tpr_rf_gp)

	print("AUC: BOOST: ",auc_boost)

	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_rf, tpr_rf, label='DART Boost')
	# plt.plot(fpr_rf_svm, tpr_rf_svm, label='SVM RBF Kernel')
	# plt.plot(fpr_rf_nn, tpr_rf_nn, label='Nearest Neighbours (k=10)')
	# plt.plot(fpr_rf_gp, tpr_rf_gp, label='Gaussian Process')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()

	ax = xgb.plot_importance(bst,importance_type='gain',max_num_features=20)


def main():	
	
	data = pd.read_csv("norm_data__non_log.txt",sep='\t').T
	label = pd.read_csv("sample_list.csv",sep=';')
	data = data.apply(np.log)

	# Conversion of string to bool
	mapping = {'Non-LCa':0,'LCa':1}
	target = label.Disease.map(mapping).values

	recursive(data.values,target)


if __name__ == "__main__":
	main()
