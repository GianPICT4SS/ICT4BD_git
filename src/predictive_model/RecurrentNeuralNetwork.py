import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import random

import sys
sys.path.insert(1, '../')

import seaborn as sns
from method_building import Prediction
from subscribe_building import InfluxDB
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import LSTM, Dense, GRU, Concatenate
from keras.engine.input_layer import Input
from keras.utils import plot_model
from keras.optimizers import Adam
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import os
from pathlib import Path

class NoModelFound(Exception):
	"""Raised when no model has been loaded"""
	pass

class EarlyStoppingAtNormDifference(tf.keras.callbacks.Callback):
	"""Stop training when the loss is at its min, i.e. the loss stops decreasing."""

	def __init__(self, patience=8):
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
			if epoch > 0:
				perc_increase = (current - best) / best * 100
				print(f'Epoch {epoch} has seen an increase of {perc_increase:.2f} % in loss metric')
			return 0
		else:
			if epoch > 0:
				perc_decrese = (best - current) / best * 100
				print(f'Epoch {epoch} has seen a decrease of {perc_decrese:.2f} % in loss metric')
			abs_diff = np.linalg.norm((best - current))
			if abs_diff < k:
				return 0
			else:
				return 1

	def on_epoch_end(self, epoch, logs=None):
		current = logs.get('val_loss')
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


class RecurrentNeuralNetwork:
	"""
		This class is used for creating a framework for predicting the Average Temperarature (1-output), the 3 Zonal Temperatures (3-output)
		and the HVAC Consumption (reverse).
		The model used is a GRU-RNN, which takes as input past data of a timeseries structured as [samples, timesteps, features]: for this purpose
		the method create_dataset is used.
		The model is created upon the Keras-Framework in the create_model method.
		The final methods are used to plot the various outputs and to estimate the differente metrics useful for selecting the best model.

		An istance of the class is created in the main.py file, where all the methods are called accordingly.

	"""
	def __init__(self, path_epout='../eplus_simulation/eplus/eplusout_Office_On_corrected_Optimal.csv', path_models='models/', path_plots='../../plots/'):
		np.random.seed(69)
		tf.random.set_seed(69)
		self.path_models = path_models
		self.path_epout = path_epout
		self.path_plots = path_plots
		self.learn = Prediction()
		self.features_considered = [
							'Temp_ext[C]', 
							'Pr_ext[Pa]', 'SolarRadiation[W/m2]', 
							'WindSpeed[m/s]','DirectSolarRadiation[W/m2]', 'AzimuthAngle[deg]', 
							'AltitudeAngle[deg]',
							'Humidity_1[%]', 'Humidity_2[%]', 'Humidity_3[%]', 
							'Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]',
							'VentAirChange_1[ach]', 'VentAirChange_2[ach]', 'VentAirChange_3[ach]',
							'Heating[J]', 'Cooling[J]'
							]

	def return_features(self):
		# influx = InfluxDB()
		# df = influx.get_dataframe()
		print('DataFrame Loaded')
		df = pd.read_csv(self.path_epout)
		df = self.learn.create_csv(df)
		features = df[self.features_considered]

		return df, features

	def create_data(self, df, features):
		minmax = sk.pipeline.Pipeline(steps=[
			('mm', preprocessing.MinMaxScaler())
			])

		quantile = sk.pipeline.Pipeline(steps=[
			('qt', preprocessing.QuantileTransformer(output_distribution='normal'))
			])

		if (self.indices) == [10,11,12]:
			self.cl = ColumnTransformer(
					[
					('qt', quantile, slice(0,13)),
					('mm', minmax, slice(13, 18)),
					],
					remainder='passthrough'
					)

		elif self.indices == [15]:
			features = features.assign(Temp_in=df[['Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]']].astype(float).mean(1))
			features = features.rename(columns={'Temp_in': 'Temp_in[C]'})
			features.pop('Temp_in1[C]')
			features.pop('Temp_in2[C]')
			features.pop('Temp_in3[C]')

			self.cl = ColumnTransformer(
					[
					('qt', quantile, slice(0,10)),
					('mm', minmax, slice(10, 15)),
					('qt2', quantile, [15])
					],
					remainder='passthrough'
					)

		elif (self.indices) == [16]:
			features = features.assign(Facilities=df['Heating[J]'] + df['Cooling[J]'])
			features = features.rename(columns={'Facilities':'Facilities[J]'})
			features.pop('Heating[J]')
			features.pop('Cooling[J]')

			self.cl = ColumnTransformer(
					[
					('qt', quantile, slice(0,13)),
					('mm', minmax, slice(13, 17)),
					# ('qt2', quantile, [14])
					],
					remainder='passthrough'
					)

		features.index = df.index
		self.index = features.index
		data = features.values
		
		self.dataset = self.cl.fit_transform(data)
		return self.dataset

	def create_dataset(self, dataset, look_back, future, step=1):
		dataX, dataY = [], []
		for i in range(0, len(dataset)-look_back-future-1, step):
			a = dataset[i:i+look_back, :]
			b = dataset[i+look_back:i+look_back+future, self.indices]
			dataX.append(a)
			dataY.append(b)

		return np.array(dataX), np.array(dataY)

	def create_train_test(self, df, features, size_train=0.8, size_val=0.1):
		dataset = self.create_data(df, features)
		N = dataset.shape[0]
		xtrain, xval, xtest = dataset[0:int(N*size_train)], dataset[int(N*size_train):int(N*(size_val + size_train))], dataset[int(N*(size_val + size_train))::]

		return xtrain, xtest, xval

	def create_model(self, df, features, look_back=24, future=6, n1=256, epochs=30, train=True, load_model=None, lr=0.001, mode='3-output', step=1, name=''):
		if mode == '3-output':
			self.indices = [10,11,12]
		elif mode == '1-output':
			self.indices = [15]
		elif mode == 'reverse':
			# want to predict Facilitites-> heating, cooling
			# need to create new feature-> sum of facilitites
			self.indices = [16]
		else:
			print('Mode can only be: reverse, 3-output, 1-output')
			exit()

		self.mode = mode
		self.name = name
		path = Path(self.path_plots)
		xtrain, xtest, xval = self.create_train_test(df, features)
		xtrain, ytrain = self.create_dataset(xtrain, look_back, future, step)
		xtest, ytest = self.create_dataset(xtest, look_back, future, step=1)
		xval, yval = self.create_dataset(xval, look_back, future, step=1)

		print(f'Train on {xtrain.shape}, validate on {xval.shape} and test on {xtest.shape}, predict for {ytrain.shape[1]}')
		#LSTM needs as input: [samples, timesteps, features]
		n_outputs = ytrain.shape[2]
		future = ytrain.shape[1]

		if train:
			batch_size = 32
			training_steps = xtrain.shape[0] // batch_size
			validation_steps = xval.shape[0] // batch_size

			inp = Input(shape=(xtrain.shape[1], xtrain.shape[2]))

			gru1 = GRU(n1, return_sequences=True, dropout=0.2)(inp)
			
			gru2 = GRU(n1//2, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(gru1)
			gru3 = GRU(n1//2, return_sequences=True, dropout=0.5, recurrent_dropout=0.4)(gru2)
			gru4 = GRU(n1//4, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)(gru3)
			# gru5 = GRU(n1//4, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)(gru4)
			# gru6 = GRU(n1//4, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)(gru5)
			gru8 = GRU(n1//4, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)(gru4)
			
			gru9 = GRU(n1//8, dropout=0.3, recurrent_dropout=0.3)(gru8)

			#output_layers: n layers x n_neurons==future (how much in the future to predict)
			out = [Dense(future)(gru9) for i in range(n_outputs)]
			model = Model(inputs=inp, outputs=out)
			
			print(model.summary())
			plot_model(model, to_file=f'{path}/Model_{self.mode}.png', show_shapes=True, dpi=300)

			model.compile(loss='mae', optimizer=Adam(learning_rate=lr))
			history = model.fit(xtrain, [ytrain[:,:,i] for i in range(n_outputs)], epochs=epochs, 
				validation_data=(xval,[yval[:,:,i] for i in range(n_outputs)]),
				batch_size=batch_size,
				callbacks=[EarlyStoppingAtNormDifference()],
				# verbose=0,
				)
			self.save_model(model, name)
			self.plot_loss(history)

		else:
			if load_model == None:
				raise NoModelFound
			else:
				model = load_model

		self.evaluate(model, xtrain, ytrain, 'Train')
		self.evaluate(model, xval, yval, 'Validation', new=0)
		self.evaluate(model, xtest, ytest, new=0)

		return model

	def load_model(self, json_name, model_name, path='models/'):
		json_file = open(path+json_name, 'r')
		json_loaded = json_file.read()

		model_loaded = model_from_json(json_loaded)
		model_loaded.load_weights(path+model_name)
		print('Model Loaded')

		return model_loaded

	def save_model(self, model, name):
		model_json = model.to_json(indent=4)
		name_json = f'models/{name}.json'
		with open(name_json, 'w+') as f:
			f.write(model_json)
		name_model = f'models/{name}.h5'
		model.save_weights(name_model)
		print('Model Saved')

	def rescale_(self, data, target, index):
		if self.mode == '1-output':
			qt = self.cl.transformers_[2][1]['qt']
			rescaled = qt.inverse_transform(target.reshape(-1,1))
			return rescaled

		elif self.mode == '3-output':
			qt1 = self.cl.transformers_[0][1]['qt']
			data = data[:,-1,:]
			data[:, index] = target
			data = data[:,0:13]
			rescaled = qt1.inverse_transform(data)
			return rescaled[:, index]

		elif self.mode == 'reverse':
			mm = self.cl.transformers_[1][1]['mm']
			data = data[:,-1,:]
			data[:, index] = target
			data = data[:, 13:]
			rescaled = mm.inverse_transform(data)
			return rescaled[:, -1]

	def evaluate(self, model, data_to_evaluate, labels, dataset='Test', new=1):
		results = np.array(model.predict(data_to_evaluate))
		if len(results.shape) == 2:
			results = results.reshape(1, results.shape[0], results.shape[1])

		path = Path(self.path_plots)
		predictions = []
		real_data = []

		for i in range(len(self.indices)):
			a = self.rescale_(data_to_evaluate, labels[:,-1,i], self.indices[i])
			b = self.rescale_(data_to_evaluate, results[i,:,-1], self.indices[i])
			predictions.append(b)
			real_data.append(a)
			
			print(dataset)
			mae = mean_absolute_error(a, b)
			r2 = r2_score(a, b, multioutput='variance_weighted')
			rmse = math.sqrt(mean_squared_error(a, b))
			print(mae, r2, rmse)
			# self.save_parameters(self.name, self.mode, dataset, mae, r2, rmse)
			plt.figure(figsize=(15,6))
			plt.plot(a[-800:], label='real_data')
			plt.plot(b[-800:], label='predictions')
			plt.legend()
			plt.grid()
			plt.xlabel('TimeSeries size')
			if self.mode == '1-output':
				plt.title('Temperature Prediction')
				plt.ylabel('Temperature[C]')
				plt.savefig(f'{path}/{self.name}_{dataset}.png')
			elif self.mode == 'reverse':
				plt.title('Consumption Prediction')
				plt.ylabel('Facility[J]')
				plt.savefig(f'{path}/{self.name}_{dataset}.png')
			else:
				plt.title(f'Temperature Zone{i} Prediction')
				plt.savefig(f'{path}/{self.name}_{i}_{dataset}.png')

			plt.close()

	def save_parameters(self, name, mode, dt, mae, r2, rmse, new):
		with open('ConfigurationModels.json') as cf:
			obj = json.loads(cf.read())

		if new:
			obj[name] = {}
		obj[name][dt] = {
			'MAE':mae,
			'R2':r2,
			'RMSE':rmse
		}

		with open('ConfigurationModels.json', 'w+') as cf:
			cf.write(json.dumps(obj, indent=4))

	def plot_loss(self, history):
		path = Path(self.path_plots)
		train_loss = history.history['loss']
		val_loss = history.history['val_loss']
		n = range(len(train_loss))
		plt.figure(figsize=(12,4))
		plt.plot(n, train_loss, label='train_loss')
		plt.plot(n, val_loss, label='validation_loss')
		plt.legend()
		plt.grid()
		plt.title('Loss Comparison')
		plt.ylabel('MAE')
		plt.xlabel('Epochs')
		plt.savefig(f'{path}/{self.name}_LossPlot.png')
		plt.close()