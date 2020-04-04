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
from sklearn.metrics import mean_squared_error
import json
import os
from pathlib import Path

np.random.seed(69)
tf.random.set_seed(69)

class TemperatureNN:
	def __init__(self, path_epout='../eplus_simulation/eplus/eplusout.csv', path_models='models/', path_plots='../../plots/'):
		self.path_epout = path_epout
		self.path_plots = path_plots
		self.learn = Prediction()
		self.features_considered = ['Temp_ext[C]', 'Pr_ext[Pa]', 'SolarRadiation[W/m2]', 'WindSpeed[m/s]',
							'DirectSolarRadiation[W/m2]', 'AzimuthAngle[deg]', 'AltitudeAngle[deg]',
							'Humidity_1[%]', 'Humidity_2[%]', 'Humidity_3[%]',
							'Heating [J]', 'Cooling [J]', 
							'Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]']

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

	def create_model(self, look_back=24, future=6, n1=50, epochs=30):
		path = Path(self.path_plots)
		xtrain, xtest, xval = self.create_train_test()

		xtrain, ytrain = self.create_dataset(xtrain, look_back, future)
		xtest, ytest = self.create_dataset(xtest, look_back, future)
		xval, yval = self.create_dataset(xval, look_back, future)
		#LSTM needs as input: [samples, timesteps, features]
		n_outputs = ytrain.shape[2]
		future = ytrain.shape[1]
		
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
		plot_model(model, to_file=f'{path}_Model.png', show_shapes=True, dpi=150)

		model.compile(loss='mae', optimizer='adam')
		model.fit(xtrain, [ytrain[:,:,i] for i in range(n_outputs)], epochs=epochs, 
			# validation_data=(xval,[yval[:,:,i] for i in range(n_outputs)]),
			# validation_steps=200, steps_per_epoch=100,
			# batch_size=12
			)

		name = f'Model_epochs{epochs}_history{look_back}_future{future}'
		self.save_model(model, name)

		self.evaluate(model, xtrain, ytrain, 'Train')
		self.evaluate(model, xtest, ytest)

		return model

	def save_model(self, model, name):
		model_json = model.to_json(indent=4)
		name_json = f'{name}.json'
		with open(name_json, 'w+') as f:
			f.write(name_json)
		name_model = f'{name}.h5'
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
			test_score = math.sqrt(mean_squared_error(a,b))
			print(f'{name} Score: %.2f RMSE' % (test_score))

			plt.figure(figsize=(15,6))
			plt.plot(a[:,-1], label='real_data')
			plt.plot(b[:,-1], label='predictions')
			plt.legend()
			plt.savefig(f'{path}/{name}_RealVSPrediction_Temp_in{i+1}[C].png')

if __name__ == '__main__':
	nn = TemperatureNN()
	past = 6
	future = 1
	name = f"model_history_{past}_ahead_{future}"
	model_name = f"{name}.h5"
	json_name = f"{name}.json"

	files = [f for f in os.listdir('.') if os.path.isfile(f)]

	nn.create_model(look_back=past, future=future)


		