import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
from src.method_building import create_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8, 6)

import tensorflow as tf

from src.method_building import Prediction

learn = Prediction()

EVALUATION_INTERVAL = 400
EPOCHS = 22
BATCH_SIZE = 256
BUFFER_SIZE = 1000




df = pd.read_csv('../eplus_simulation/eplus/eplusout.csv')
df = create_csv(df)
TRAIN_SPLIT = int(df.shape[0]*0.8)
tf.random.set_seed(123)


# =================================================================
# Recurrent Neural Network(RNN): Long Short Term Memory(LSTM)
# =================================================================


"""
# ============================================
# Part 2: Forecast a multivariate TimeSeries

In a multi-step prediction model, given a past history, the model needs to learn to predict a range of future values.
Thus, unlike a single step model, where only a single future point is predicted, a multi-step model predict a sequence 
of the future. For the multi-step model, the training data again consists of recordings over the past five days sampled
every hour. However, here, the model needs to learn to predict the temperature for the next 12 hours. 
Since an observation is taken every 10 minutes, the output is 72 predictions. For this task, the dataset needs to be 
prepared accordingly, thus the first step is just to create it again, but with a different target window.
# ============================================
"""

features_considered = ['Temp_ext[C]', 'Pr_ext[Pa]', 'SolarRadiation[W/m2]',
                       'InfHeatLoss_2[J]', 'InfAirChange_2[ach]', 'InfHeatLoss_3[J]',
                       'InfAirChange_3[ach]', 'InfHeatLoss_1[J]', 'InfAirChange_1[ach]',
                       'VentHeatLoss_2[J]', 'VentAirChange_2[ach]', 'VentHeatLoss_3[J]',
                       'VentAirChange_3[ach]', 'VentHeatLoss_1[J]', 'VentAirChange_1[ach]',
                       'Humidity_1[%]', 'Humidity_2[%]', 'Humidity_3[%]'
    ]

features = df[features_considered]
features['Temp_in[C]'] = df[['Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]']].astype(float).mean(1)
#features['Humidity_in[%]'] = df[['Humidity_1[%]', 'Humidity_2[%]', 'Humidity_3[%]']].astype(float).mean(1)
features.index = df.index


dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std


past_history = 24*10  # 10 days history
future_target = 24  # one-day prediction
STEP = 1

x_train_multi, y_train_multi = learn.multivariate_data(dataset=dataset, target=dataset[:, -1], start_index=0,
                                                 end_index=TRAIN_SPLIT, history_size=past_history,
                                                 target_size=future_target, step=STEP)
x_val_multi, y_val_multi = learn.multivariate_data(dataset=dataset, target=dataset[:, -1],
                                             start_index=TRAIN_SPLIT, end_index=None, history_size=past_history,
                                             target_size=future_target, step=STEP)

print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


#leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          dropout=0.2,
                                          recurrent_dropout=0.5,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16,
                                          dropout=0.2,
                                          recurrent_dropout=0.5,
                                          activation='relu'))
                                          #kernel_initializer='he_normal'))

multi_step_model.add(tf.keras.layers.Dense(future_target))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0008, clipvalue=1.0), loss='mae')


multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=150)

learn.plot_train_history(multi_step_history, 'lr_0.0008')

for x, y in val_data_multi.take(3):
    learn.multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])




# whith generator
"""
lookback = 24*10
steps = 1
delay = 24
batch_size = 256

train_gen = learn.generator(dataset, lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=TRAIN_SPLIT,
                      shuffle=True,
                      step=steps,
                      batch_size=batch_size)

val_gen = learn.generator(dataset, lookback=lookback,
                      delay=delay,
                      min_index=6001,
                      max_index=7008,
                      step=steps,
                      batch_size=batch_size)

test_gen = learn.generator(dataset, lookback=lookback,
                      delay=delay,
                      min_index=7008,
                      max_index=None,
                      step=steps,
                      batch_size=batch_size)


val_steps = (7008 - 6001 - lookback)
test_steps = (int(df.shape[0]) - 7008 - lookback)


leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.GRU(32,
                                          dropout=0.2,
                                          recurrent_dropout=0.5,
                                          return_sequences=True,
                                          input_shape=(None, dataset.shape[-1])))
multi_step_model.add(tf.keras.layers.GRU(16,
                                          dropout=0.2,
                                          recurrent_dropout=0.5,
                                          activation=leakly_relu,
                                         kernel_initializer='he_normal'))
multi_step_model.add(tf.keras.layers.Dense(24))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0, learning_rate=0.0001), loss='mae')


multi_step_history = multi_step_model.fit_generator(train_gen, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_gen,
                                          validation_steps=50)

learn.plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_gen.take(3):
    learn.multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

acc = multi_step_history.history['acc']
val_acc = multi_step_history.history['val_acc']
loss = multi_step_history.history['loss']
val_loss = multi_step_history.history['val_loss']

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

"""




""" dropout-regularized Stacked GRU model
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




