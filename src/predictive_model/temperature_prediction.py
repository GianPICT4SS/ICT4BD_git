import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
from src.method_building import create_csv, generator
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


df = pd.read_csv('../eplus_simulation/eplus/eplusout.csv')
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
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, df.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps
                              )

""" dropout-regularized Stacked GRU model
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
                              steps_per_epoch=200,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
ax1.plot(epochs, acc, 'bo', label='Training acc')
ax1.plot(epochs, val_acc, 'b', label='Validation acc')
ax1.set_title('Training and validation accuracy')
ax1.legend()
ax1.grid(linestyle='--', linewidth=.4, which='both')

ax2.plot(epochs, loss, 'bo', label='Training loss')
ax2.plot(epochs, val_loss, 'b', label='Validation loss')
ax2.set_title('Training and validation loss')
ax2.legend()
ax2.grid(linestyle='--', linewidth=.4, which='both')
plt.subplots_adjust(bottom=0.4, right=0.8, top=0.9, hspace=1)
plt.savefig(fname='../../plots/prediction_model_error.png', dpi=400)
plt.close()

