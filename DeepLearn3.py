import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.metrics import recall_score, classification_report, auc, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import optimizers
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NearMiss
from keras import backend as K

# 100 original
seed = 200
np.random.seed(seed)

def lr_select(data,target):
	from sklearn.linear_model import LogisticRegression

	model = LogisticRegression().fit(data,target)

	from sklearn.feature_selection import SelectFromModel

	selection = SelectFromModel(model,max_features=60,prefit=True)
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

	print(DATA.shape)

	label = read_csv("sample_list.csv",sep=';')
	DATA = DATA.apply(np.log).values # Retain the log due to the maximising values

	# Conversion of string to bool
	mapping = {'Non-LCa':0,'LCa':1}
	TARGET = label.Disease.map(mapping).values

	DATA = lr_select(DATA,TARGET)
	print(DATA.shape)

	class_weight = {1:4,0:1}

	kf = KFold(n_splits=3, random_state=seed, shuffle=True)
	acc = []
	prec = []
	recall = []
	auc = []
	with open('results_deepl3.txt', 'w') as f:
		for train_index, test_index in kf.split(DATA):
			X_train, X_test, y_train, y_test = DATA[train_index],DATA[test_index],TARGET[train_index],TARGET[test_index]
			
			from sklearn.preprocessing import MinMaxScaler
			scaler = MinMaxScaler().fit(X_train)
			X_train = scaler.transform(X_train)
			X_test = scaler.transform(X_test)


			nb_epoch = 150
			batch_size = 256
			input_dim = DATA.shape[1]
			learning_rate = 1e-3
			decay = learning_rate/nb_epoch

			input_layer = Input(shape=(input_dim, ))

			net = Dense(input_dim,activation="linear",activity_regularizer=regularizers.l1(learning_rate))(input_layer)
			net = Dense(15,activation="linear")(net)
			net = Dense(10,activation="linear")(net)
			net = Dense(5,activation="linear")(net)

			output_layer = Dense(1, activation='sigmoid')(net)

			adam = optimizers.Adam(lr=learning_rate,decay=decay)

			model = Model(inputs=input_layer, outputs=output_layer)
			model.compile(metrics=['accuracy',precision_m,recall_m,f1_m],
		                    loss='binary_crossentropy',
		                    optimizer=adam)


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
			                    class_weight=class_weight,
			                    callbacks=[cp, tb]).history


			# This is to figure out the correct number of epochs before training
			# becomes redundant. Here, it is discovered 80 epochs satisfies this problem
			# Uncomment the code to visualise the test v train loss plot. 
			plt.plot(history['loss'], linewidth=2, label='Train')
			plt.plot(history['val_loss'], linewidth=2, label='Test')
			plt.legend(loc='upper right')
			plt.title('Model loss')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			#plt.ylim(ymin=0.70,ymax=1)
			plt.show()

			# load weights
			model.load_weights("NeuralNetworkModel.h5")
			# Compile model (required to make predictions)
			model.compile(metrics=['accuracy',precision_m,recall_m,f1_m],
		                    loss='binary_crossentropy',
		                    optimizer=adam)
			y_pred = model.predict(X_test)


			plt.style.use(['seaborn-colorblind'])
			fpr, tpr, _ = roc_curve(y_test, y_pred)
			plt.figure(1)
			plt.plot([0, 1], [0, 1], 'k--')
			plt.plot(fpr, tpr, label='Deep Net',alpha=0.6,color='r')
			plt.xlabel('False positive rate')
			plt.ylabel('True positive rate')
			plt.title('ROC curve AUC = {}'.format(roc_auc_score(y_test, y_pred)))
			plt.legend(loc='best')
			plt.show()

			auc.append(roc_auc_score(y_test, y_pred))
			recall.append(recall_score(y_test,np.round(y_pred,0)))
			prec.append(precision_score(y_test,np.round(y_pred,0)))
			acc.append(accuracy_score(y_test,np.round(y_pred,0)))

			# AVERAGE PRECISION

			from sklearn.metrics import average_precision_score
			average_precision = average_precision_score(y_test, y_pred)

			print('Average precision-recall score: {0:0.2f}'.format(
			      average_precision))

			# PRECISION RECALL CURVE
			from sklearn.metrics import precision_recall_curve
			
			from inspect import signature

			precision, recall1, _ = precision_recall_curve(y_test, y_pred)

			# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
			step_kwargs = ({'step': 'post'}
			               if 'step' in signature(plt.fill_between).parameters
			               else {})
			plt.step(recall1, precision, color='b', alpha=0.2,
			         where='post')
			plt.fill_between(recall1, precision, alpha=0.2, color='b', **step_kwargs)

			plt.xlabel('Recall')
			plt.ylabel('Precision')
			plt.ylim([0.0, 1.05])
			plt.xlim([0.0, 1.0])
			plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
			          average_precision))
			plt.show()



		print(np.mean(auc),np.mean(recall),np.mean(prec),np.mean(acc))
		print("MODEL 9 Hidden, Hidden Nodes [200,400,600,800,1000,800,600,400,200], \n L2 Regulariser Layer 1, Epoch: 300, LearnRate: 1e-20, Loss: Binary Cross, Opt: ADAM \n CV: 5 \n\n\n",file=f)
		print('N_FEATURES: {}, AUC: {}, RECALL: {}, PRECISION: {}, ACCURACY: {} \n\n\n'.format(DATA.shape[1],auc,recall,prec,acc),file=f)
		print('AUC MEAN: {}\n RECALL MEAN: {}\n PRECISON MEAN: {}\n ACCURACY MEAN: {}'.format(np.mean(auc),np.mean(recall),np.mean(prec),np.mean(acc)))
		






if __name__ == '__main__':
	main()






