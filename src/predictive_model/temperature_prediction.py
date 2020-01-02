# Python >3.4 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn > 0.19 is required
#import sklearn
from sklearn import preprocessing
#assert sklearn.__version__ >= "0.20"

# TensorFlow >= 2.0 is required
import tensorflow as tf


assert tf.__version__ >= "2.0"

if not tf.test.is_gpu_available():
    print("No GPU was detected: LSTMs and CNNs can be very slow without a GPU.")

# Common imports
import numpy as np
import pandas as pd
import os

# to make this script's output stable across runs
np.random.seed(123)
tf.random.set_seed(123)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To use some specific method useful for this task
from src.method_building import Prediction


learn = Prediction()
# ===========================================================
# PARAMETERS
# ===========================================================
EVALUATION_INTERVAL = 270
EPOCHS = 20
BATCH_SIZE = 156
BUFFER_SIZE = 1000

# ==============================================================
# DATASET
# ===============================================================
df_ = pd.read_csv('../eplus_simulation/eplus/eplusout.csv')
df = learn.create_csv(df_)
TRAIN_SPLIT = int(df.shape[0]*0.8)  # 80% data train

# =================================================================
# Recurrent Neural Network(RNN): Long Short Term Memory(LSTM)
# =================================================================



features_considered = ['Temp_ext[C]', 'Pr_ext[Pa]', 'SolarRadiation[W/m2]', 'WindSpeed[m/s]', 'DirectSolarRadiation[W/m2]',
                       'InfHeatLoss_2[J]', 'InfAirChange_2[ach]', 'InfHeatLoss_3[J]',
                       'InfAirChange_3[ach]', 'InfHeatLoss_1[J]', 'InfAirChange_1[ach]',
                       'VentHeatLoss_2[J]', 'VentAirChange_2[ach]', 'VentHeatLoss_3[J]',
                       'VentAirChange_3[ach]', 'VentHeatLoss_1[J]', 'VentAirChange_1[ach]',
                       'Humidity_1[%]', 'Humidity_2[%]', 'Humidity_3[%]',

    ]

features = df[features_considered]
#features['Temp_in[C]'] = df[['Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]']].astype(float).mean(1).copy()
features = features.assign(Temp_in=df[['Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]']].astype(float).mean(1))
features = features.rename(columns={'Temp_in': 'Temp_in[C]'})
features.index = df.index

# Standardization data
dataset = features.values
#scaler = preprocessing.StandardScaler().fit(dataset)
#dataset_scaled = scaler.transform(dataset)
dataset_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
dataset_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-dataset_mean)/dataset_std

past_history = 24*10  # 10 days history
future_target = 24  # one-day prediction
STEP = 1



x_train_multi, y_train_multi = learn.multivariate_data(dataset=dataset, target=dataset[:, -1], start_index=0,
                                                 end_index=TRAIN_SPLIT, history_size=past_history,
                                                 target_size=future_target, step=STEP)
x_val_multi, y_val_multi = learn.multivariate_data(dataset=dataset, target=dataset[:, -1],
                                             start_index=(TRAIN_SPLIT - int(TRAIN_SPLIT*0.4)), end_index=(TRAIN_SPLIT), history_size=past_history,
                                             target_size=future_target, step=STEP)
x_test_multi, y_test_multi = learn.multivariate_data(dataset=dataset, target=dataset[:, -1],
                                             start_index=(TRAIN_SPLIT+1), end_index=None, history_size=past_history,
                                             target_size=future_target, step=STEP)

print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('Target temperature to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(64,
                                          dropout=0.1,
                                          recurrent_dropout=0.3,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
#multi_step_model.add(tf.keras.layers.LSTM(64,
#                                          dropout=0.1,
#                                          recurrent_dropout=0.3,
#                                          activation='relu',
#                                          return_sequences=True))
                                          #kernel_initializer='he_normal'))
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          dropout=0.1,
                                          recurrent_dropout=0.3,
                                          activation='relu'))
                                          #kernel_initializer='he_normal'))

multi_step_model.add(tf.keras.layers.Dense(future_target))

print(multi_step_model.summary())


#multi_step_model.compile(optimizer=tf.keras.optimizers.Nadam(), loss='mae')
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='mae',
                         metrics=['accuracy'])


multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=150,
                                          batch_size=68)

#learn.plot_train_history(multi_step_history, 'NAdam')
learn.plot_train_history(multi_step_history, 'RMSprop_lr_0.0001')

for x, y in val_data_multi.take(3):
    learn.multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

y_pred_train = multi_step_model.predict(x_train_multi)

y_pred_test = multi_step_model.predict(x_test_multi)


for i in range(3):
    learn.multi_step_plot(x_test_multi[i], y_test_multi[i], y_pred_test[i])



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

leakly_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.GRU(32,
                                          dropout=0.1,
                                          recurrent_dropout=0.5,
                                          return_sequences=True,
                                          input_shape=(None, dataset.shape[-1])))

multi_step_model.add(tf.keras.layers.GRU(64,
                                          dropout=0.1,
                                          recurrent_dropout=0.5,
                                          activation=leakly_relu,
                                         kernel_initializer='he_normal'))
multi_step_model.add(tf.keras.layers.Dense(24))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0, learning_rate=0.0001), loss='mae')


multi_step_history = multi_step_model.fit_generator(train_gen, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_gen,
                                          validation_steps=val_steps)

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



