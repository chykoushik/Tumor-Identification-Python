import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.metrics import recall_score, classification_report, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NearMiss
from keras import backend as K

# 100 original
seed = 100
np.random.seed(seed)

def boost_select(data,target):

	import xgboost as xgb
	import sklearn as sk

	X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(data,target,shuffle=True,test_size=0.33)
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

	# This code is to show how many features surpass a good limit, minimum number = 42
 	# By generating the code below, we can see the fit of the model given number of components.
 	# Here, we simply waited for the minimum number of components to reach 0.85 threshold
 	# then using these components to optimise the deep learning model below. 
 	# Uncomment the code below to see the feature importance with their governing thresholds. 

	# thresholds = sort(model.feature_importances_)[::-1]
	# print(thresholds)
	# for thresh in thresholds:
	#     # select features using threshold
	#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
	#     select_X_train = selection.transform(X_train_)
	#     # train model
	#     selection_model = xgb.XGBClassifier(params=param)
	#     selection_model.fit(select_X_train, y_train_)
	#     # eval model
	#     select_X_test = selection.transform(X_test)
	#     y_pred = selection_model.predict(select_X_test)
	#     predictions = [round(value) for value in y_pred]
	#     accuracy = sk.metrics.accuracy_score(y_test, predictions)
	#     recall = sk.metrics.recall_score(y_test,predictions)
	#     precision = sk.metrics.precision_score(y_test,predictions)
	#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%%%, Recall: %.2f%%%%, Precision: %.2f%%%%" % (thresh, select_X_train.shape[1], accuracy*100.0,recall*100.0,precision*100.0))
	    
	selection = SelectFromModel(model,max_features=42,prefit=True)
	new_data = selection.transform(data)
	return new_data


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def main():
	from pandas import read_csv
	# Read in the data

	# # NORMAL DATA
	# DATA = read_csv("norm_data__non_log.txt",sep='\t').T
	# DATA = DATA.apply(np.log).values # Retain the log due to the maximising values
	
	# MIN MAX DATA
	DATA = read_csv("norm_data__non_log.txt",sep='\t').T
	label = read_csv("sample_list.csv",sep=';')
	DATA = DATA.apply(np.log).values # Retain the log due to the maximising values

	# Conversion of string to bool
	mapping = {'Non-LCa':0,'LCa':1}
	TARGET = label.Disease.map(mapping).values

	print(DATA.shape)

	DATA = boost_select(DATA,TARGET)

	kf = KFold(n_splits=5, random_state=seed, shuffle=True)
	acc = []
	prec = []
	recall = []
	auc = []
	with open('results_deep.txt', 'w') as f:
		for train_index, test_index in kf.split(DATA):
			X_train, X_test, y_train, y_test = DATA[train_index],DATA[test_index],TARGET[train_index],TARGET[test_index]
			
			from sklearn.preprocessing import MinMaxScaler
			scaler = MinMaxScaler().fit(X_train)
			X_train = scaler.transform(X_train)
			X_test = scaler.transform(X_test)

			ada = ADASYN()
			X_train, y_train = ada.fit_resample(X_train,y_train)

			nb_epoch = 80
			batch_size = 64
			input_dim = DATA.shape[1] 
			learning_rate = 1e-7

			input_layer = Input(shape=(input_dim, ))

			net = Dense(200,activation="relu",activity_regularizer=regularizers.l2(learning_rate))(input_layer)
			net = Dense(400, activation="relu")(net)
			net = Dense(600, activation="relu")(net)
			net = Dense(800, activation="relu")(net)
			net = Dense(1000,activation="relu")(net)
			net = Dense(800,activation="relu")(net)
			net = Dense(600, activation="relu")(net)
			net = Dense(400, activation="relu")(net)
			net = Dense(200, activation="relu")(net)



			output_layer = Dense(1, activation='sigmoid')(net)

			
			model = Model(inputs=input_layer, outputs=output_layer)
			model.compile(metrics=['accuracy',precision_m,recall_m,f1_m],
		                    loss='binary_crossentropy',
		                    optimizer='adam')


			cp = ModelCheckpoint(filepath="NeuralNetworkModel.h5",
			                               save_best_only=True,
			                               verbose=0)

			tb = TensorBoard(log_dir='./logs',
			                histogram_freq=0,
			                write_graph=True,
			                write_images=True)

			history = model.fit(X_train, y_train,
			                    epochs=nb_epoch,
			                    batch_size=batch_size,
			                    shuffle=True,
			                    validation_data=(X_test, y_test),
			                    verbose=1,
			                    callbacks=[cp, tb]).history


			# This is to figure out the correct number of epochs before training
			# becomes redundant. Here, it is discovered 80 epochs satisfies this problem
			# Uncomment the code to visualise the test v train loss plot. 
			# plt.plot(history['loss'], linewidth=2, label='Train')
			# plt.plot(history['val_loss'], linewidth=2, label='Test')
			# plt.legend(loc='upper right')
			# plt.title('Model loss')
			# plt.ylabel('Loss')
			# plt.xlabel('Epoch')
			# #plt.ylim(ymin=0.70,ymax=1)
			# plt.show()

			# load weights
			model.load_weights("NeuralNetworkModel.h5")
			# Compile model (required to make predictions)
			model.compile(metrics=['accuracy',precision_m,recall_m,f1_m],
		                    loss='binary_crossentropy',
		                    optimizer='adam')
			y_pred = model.predict(X_test)

			auc.append(roc_auc_score(y_test, y_pred))
			recall.append(recall_score(y_test,np.round(y_pred,0)))
			prec.append(precision_score(y_test,np.round(y_pred,0)))
			acc.append(accuracy_score(y_test,np.round(y_pred,0)))

		print(np.mean(auc),np.mean(recall),np.mean(prec),np.mean(acc))
		print("MODEL 9 Hidden, Hidden Nodes [200,400,600,800,1000,800,600,400,200], \n L2 Regulariser Layer 1, Epoch: 80, LearnRate: 1e-7, Loss: Binary Cross, Opt: ADAM \n CV: 5 \n\n\n",file=f)
		print('N_FEATURES: {}, AUC: {}, RECALL: {}, PRECISION: {}, ACCURACY: {}, '.format(42,auc,recall,prec,acc),file=f)







if __name__ == '__main__':
	main()






