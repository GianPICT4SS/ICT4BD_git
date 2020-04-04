import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
import sys
sys.path.insert(1, '../')
from method_building import Prediction
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model, model_from_json
from keras.layers import LSTM, Dense, GRU
from keras.engine.input_layer import Input
from keras.utils import plot_model
import math
from sklearn.metrics import r2_score
import json
import os
from pathlib import Path

np.random.seed(69)
tf.random.set_seed(69)

class NoModelFound(Exception):
	"""Raised when no model has been loaded"""
	pass

class EarlyStoppingAtNormDifference(tf.keras.callbacks.Callback):
	"""Stop training when the loss is at its min, i.e. the loss stops decreasing."""

	def __init__(self, patience=0):
		super(EarlyStoppingAtNormDifference, self).__init__()

		self.patience = patience

		# best_weights to store the weights at which the minimum loss occurs.
		self.best_weights = None

	def on_train_begin(self, logs=None):
		# The number of epoch it has waited when loss is no longer minimum.
		self.wait = 0
		# The epoch the training stops at.
		self.stopped_epoch = 0
		# Initialize the best as infinity.
		self.best = np.Inf

	def check_condition(self, current, best, epoch, k=0.001):
		if current > best:
			perc_increase = (current - best) / best * 100
			print(f'Epoch {epoch} has seen an increase of {perc_increase:.2f} % in loss metric')
		else:
			perc_decrese = (best - current) / best * 100
			print(f'Epoch {epoch} has seen a decrease of {perc_decrese:.2f} % in loss metric')
			abs_diff = np.linalg.norm((best - current))
			if abs_diff <= k:
				return 0
			else:
				return 1

	def on_epoch_end(self, epoch, logs=None):
		current = logs.get('loss')
		if self.check_condition(current, self.best, epoch):
		  self.best = current
		  self.wait = 0
		  # Record the best weights if current results is better (less).
		  self.best_weights = self.model.get_weights()
		else:
		  self.wait += 1
		  if self.wait >= self.patience:
		    self.stopped_epoch = epoch
		    self.model.stop_training = True
		    print('Restoring model weights from the end of the best epoch.')
		    self.model.set_weights(self.best_weights)

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
		  print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


class TemperatureNN:
	def __init__(self, path_epout='../eplus_simulation/eplus/eplusout.csv', path_models='models/', path_plots='../../plots/'):
		self.path_models = path_models
		self.path_epout = path_epout
		self.path_plots = path_plots
		self.learn = Prediction()
		self.features_considered = [
							'Temp_ext[C]', 'Pr_ext[Pa]', 'SolarRadiation[W/m2]', 
							'WindSpeed[m/s]','DirectSolarRadiation[W/m2]', 'AzimuthAngle[deg]', 
							'AltitudeAngle[deg]','Humidity_1[%]', 'Humidity_2[%]', 
							'Humidity_3[%]', 'Heating [J]', 'Cooling [J]', 
							'Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]'
							]

		self.indices = [12,13,14]

	def create_data(self):
		df = pd.read_csv(self.path_epout)
		df = self.learn.create_csv(df)

		features = df[self.features_considered]
		features.index = df.index
		data = features.values

		scaler_data = preprocessing.StandardScaler().fit(data)
		dataset = scaler_data.transform(data)

		self.means = [scaler_data.mean_[i] for i in self.indices]
		self.stds = [scaler_data.scale_[i] for i in self.indices]

		return dataset

	def create_dataset(self, dataset, look_back, future):
		dataX, dataY = [], []
		for i in range(len(dataset)-look_back-future-1):
			a = dataset[i:(i+look_back), :]
			b = dataset[(i + look_back):(i + look_back + future), self.indices]
			dataX.append(a)
			dataY.append(b)

		return np.array(dataX), np.array(dataY)

	def create_train_test(self, size_train=0.7, size_val=0.1):
		dataset = self.create_data()
		N = dataset.shape[0]
		size_test = size_train + size_val
		xtrain, xtest, xval = dataset[0:int(N*size_train)], dataset[int(N*size_train):int(N*size_test)], dataset[int(N*size_test)::]

		return xtrain, xtest, xval

	def create_model(self, look_back=24, future=6, n1=50, epochs=30, train=True, load_model=None):
		path = Path(self.path_plots)
		xtrain, xtest, xval = self.create_train_test()

		xtrain, ytrain = self.create_dataset(xtrain, look_back, future)
		xtest, ytest = self.create_dataset(xtest, look_back, future)
		xval, yval = self.create_dataset(xval, look_back, future)
		#LSTM needs as input: [samples, timesteps, features]
		n_outputs = ytrain.shape[2]
		future = ytrain.shape[1]
		if train:
			inp = Input(shape=(xtrain.shape[1], xtrain.shape[2]))

			gru1 = GRU(n1, return_sequences=True)(inp)
			gru2 = GRU(n1//2, return_sequences=True, activation='relu', dropout=0.1)(gru1)
			gru3 = GRU(n1//4, activation='relu', dropout=0.05)(gru2)

			#output_layers: 3 layers (3 temperatures) x n_neuros==future (how much in the future to predict)
			d1 = Dense(future)(gru3)
			d2 = Dense(future)(gru3)
			d3 = Dense(future)(gru3)

			model = Model(inputs=inp, outputs=[d1,d2,d3])
			
			print(model.summary())
			plot_model(model, to_file=f'{path}/Model.png', show_shapes=True, dpi=300)

			model.compile(loss='mae', optimizer='adam')
			history = model.fit(xtrain, [ytrain[:,:,i] for i in range(n_outputs)], epochs=epochs, 
				# validation_data=(xval,[yval[:,:,i] for i in range(n_outputs)]),
				# validation_steps=200, steps_per_epoch=100,
				# batch_size=12,
				callbacks=[EarlyStoppingAtNormDifference()]
				)
			name = f'Model_epochs{epochs}_history{look_back}_future{future}'
			self.save_model(model, name)
			self.plot_loss(history)

		else:
			if load_model == None:
				raise NoModelFound
			else:
				model = load_model

		self.evaluate(model, xtrain, ytrain, 'Train')
		self.evaluate(model, xtest, ytest)

		return model

	def save_model(self, model, name):
		model_json = model.to_json(indent=4)
		name_json = f'models/{name}.json'
		with open(name_json, 'w+') as f:
			f.write(model_json)
		name_model = f'models/{name}.h5'
		model.save_weights(name_model)
		print('Model Saved')

	def evaluate(self, model, data_to_evaluate, labels, name='Test'):
		results = np.array(model.predict(data_to_evaluate))
		path = Path(self.path_plots)
		predictions = []
		real_data = []
		for i in range(results.shape[0]):
			a = labels[:,:,i]*self.stds[i] + self.means[i]
			b = results[i,:,:]*self.stds[i] + self.means[i]
			predictions.append(b)
			real_data.append(a)
			
			test_score = r2_score(b, a, multioutput='variance_weighted')
			print(f'{name} R2 Score: %.2f (1=Best Possible Model)' % (test_score))

			plt.figure(figsize=(15,6))
			plt.plot(a[:,-1], label='real_data')
			plt.plot(b[:,-1], label='predictions')
			plt.legend()
			plt.grid()
			plt.title(f'{name}_RealVSPrediction_Temp_in{i+1}[C]')
			plt.xlabel('TimeSeries size')
			plt.ylabel('Temperature [C]')
			plt.savefig(f'{path}/{name}_RealVSPrediction_Temp_in{i+1}[C].png')

	def load_model(self, json_name, model_name, path='models/'):
		json_file = open(path+json_name, 'r')
		json_loaded = json_file.read()

		model_loaded = model_from_json(json_loaded)
		model_loaded.load_weights(path+model_name)
		print('Model Loaded')

		return model_loaded

	def plot_loss(self, history):
		path = Path(self.path_plots)
		losses = history.history['loss']
		plt.figure(figsize=(12,4))
		plt.plot(losses)
		plt.grid()
		plt.title('Train Loss')
		plt.ylabel('MAE')
		plt.xlabel('Epochs')
		plt.savefig(f'{path}/TrainLoss.png')

if __name__ == '__main__':
	nn = TemperatureNN()
	past = 24
	future = 3
	epochs = 10

	name = f"Model_epochs{epochs}_history{past}_future{future}"
	model_name = f"{name}.h5"
	json_name = f"{name}.json"

	models = os.listdir('models/')
	if json_name not in models:	
		nn.create_model(look_back=past, future=future, epochs=epochs)
	else:
		model = nn.load_model(json_name, model_name)
		nn.create_model(look_back=past, future=future, epochs=epochs, train=0, load_model=model)

		