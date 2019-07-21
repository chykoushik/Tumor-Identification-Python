import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.metrics import recall_score

from keras import backend as K

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
	DATA = read_csv("density_featureselect_003.csv")
	label = read_csv("sample_list.csv",sep=';')
	DATA.set_index('Unnamed: 0',inplace=True)
	DATA = DATA.apply(np.log).values # Retain the log due to the maximising values

	# Conversion of string to bool
	mapping = {'Non-LCa':0,'LCa':1}
	TARGET = label.Disease.map(mapping).values

	print(DATA.shape)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(DATA,TARGET,test_size=0.10,shuffle=True)

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	ada = ADASYN()
	X_train, y_train = ada.fit_resample(X_train,y_train)

	nb_epoch = 220
	batch_size = 256
	input_dim = DATA.shape[1] 
	learning_rate = 1e-9

	input_layer = Input(shape=(input_dim, ))

	net = Dense(200, activation="relu",activity_regularizer=regularizers.l1(learning_rate))(input_layer)
	net = Dense(600, activation="relu")(net)
	net = Dense(800, activation="relu")(net)
	net = Dense(1000, activation="relu")(net)
	net = Dense(800, activation="relu")(net)
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


	plt.plot(history['loss'], linewidth=2, label='Train')
	plt.plot(history['val_loss'], linewidth=2, label='Test')
	plt.legend(loc='upper right')
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	#plt.ylim(ymin=0.70,ymax=1)
	plt.show()



if __name__ == '__main__':
	main()






