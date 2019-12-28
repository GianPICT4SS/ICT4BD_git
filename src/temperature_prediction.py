import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
from method_building import create_csv, generator
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


df = pd.read_csv('eplus/eplusout.csv')
df = create_csv(df)
TRAIN = int(df.shape[0]*0.8)
# parsing the data
float_data = df.values
# Normalizing the data
X_train = df[:TRAIN]
mean = float_data[:TRAIN].mean(axis=0)
std = float_data[:TRAIN].std(axis=0)
float_data_std = (float_data - mean)/std

lookback = 24*30
steps = 1
delay = 24
batch_size = 128

train_gen = generator(float_data_std, lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=6000,
                      shuffle=True,
                      step=steps,
                      batch_size=batch_size)

val_gen = generator(float_data_std, lookback=lookback,
                      delay=delay,
                      min_index=6001,
                      max_index=7000,
                      step=steps,
                      batch_size=batch_size)

test_gen = generator(float_data_std, lookback=lookback,
                      delay=delay,
                      min_index=7001,
                      max_index=None,
                      step=steps,
                      batch_size=batch_size)


val_steps = (7000 - 6001 - lookback)
test_steps = (int(df.shape[0]) - 70001 - lookback)

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, df.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
