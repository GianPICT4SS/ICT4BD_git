# Python >3.4 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn > 0.19 is required
import sklearn
from sklearn import preprocessing


import tensorflow as tf

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
EVALUATION_INTERVAL = 500
EPOCHS = 20
BATCH_SIZE = 56
BUFFER_SIZE = 5000

# ==============================================================
# DATASET
# ===============================================================
df = pd.read_csv('../eplus_simulation/eplus/eplusout.csv')
df = learn.create_csv(df)
TRAIN_SPLIT = int(df.shape[0]*0.8)  # 80% data train

# =================================================================
# Recurrent Neural Network(RNN): Long Short Term Memory(LSTM)
# =================================================================



features_considered = ['Temp_ext[C]', 'Pr_ext[Pa]', 'SolarRadiation[W/m2]', 'WindSpeed[m/s]',
                       'DirectSolarRadiation[W/m2]', 'AzimuthAngle[deg]', 'AltitudeAngle[deg]',
                       'Humidity_1[%]', 'Humidity_2[%]', 'Humidity_3[%]',
                       'Heating [J]', 'Cooling [J]'

    ]

features = df[features_considered]
#features['Temp_in[C]'] = df[['Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]']].astype(float).mean(1).copy()
features = features.assign(Temp_in=df[['Temp_in1[C]', 'Temp_in2[C]', 'Temp_in3[C]']].astype(float).mean(1))
features = features.rename(columns={'Temp_in': 'Temp_in[C]'})
features.index = df.index


# Standardization data
target = features.pop('Temp_in[C]').values
target = target.reshape(target.shape[0], 1)
dataset = features.values

scaler_ds = preprocessing.StandardScaler().fit(dataset)
scaler_tar = preprocessing.StandardScaler().fit(target)
dataset = scaler_ds.transform(dataset)
target = scaler_tar.transform(target)
#dataset_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
#dataset_std = dataset[:TRAIN_SPLIT].std(axis=0)
#dataset = (dataset-dataset_mean)/dataset_std

past_history = 24*10  # 10 days history
future_target = 24  # one-day prediction
STEP = 1



x_train_multi, y_train_multi = learn.multivariate_data(dataset=dataset, target=target, start_index=0,
                                                 end_index=TRAIN_SPLIT, history_size=past_history,
                                                 target_size=future_target, step=STEP)
x_val_multi, y_val_multi = learn.multivariate_data(dataset=dataset, target=target,
                                             start_index=TRAIN_SPLIT, end_index=None, history_size=past_history,
                                             target_size=future_target, step=STEP)




print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('Target temperature to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE, reshuffle_each_iteration=False).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# ======================================================
# Building the LSTM Networks
# ======================================================

leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.GRU(91,
                                          dropout=0.05,
                                          recurrent_dropout=0.2,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.GRU(65,
                                          dropout=0.05,
                                          recurrent_dropout=0.2,
                                          activation='relu',
                                          return_sequences=True))
                                          #kernel_initializer='he_normal'))
multi_step_model.add(tf.keras.layers.GRU(39,
                                          dropout=0.05,
                                          recurrent_dropout=0.2,
                                          activation='relu'))
                                          #kernel_initializer='he_normal'))

multi_step_model.add(tf.keras.layers.Dense(future_target))

print(multi_step_model.summary())


#multi_step_model.compile(optimizer=tf.keras.optimizers.Nadam(), loss='mae', metrics=['accuracy'])
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mae',
                         metrics=['accuracy'])


multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

results = multi_step_model.evaluate(x_val_multi, y_val_multi)
print('Test Acc.: {:.2f}%'.format(results[1]*100))

#learn.plot_train_history(multi_step_history, 'NAdam')
learn.plot_train_history(multi_step_history, 'RMSprop_lr_0.001')

for x, y in val_data_multi.take(3):
    y_ = multi_step_model.predict(x)
    y_ = scaler_tar.inverse_transform(y_)
    x = scaler_ds.inverse_transform(x[0])
    y = scaler_tar.inverse_transform(y[0])
    learn.multi_step_plot(x, y, y_[0], model='GRU_dropout')

def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(y_true - y_pred)

y_pred_test = multi_step_model.predict(x_val_multi)
y_pred_train = multi_step_model.predict(x_train_multi)

y_pred_train = scaler_tar.inverse_transform(y_pred_train)
y_pred_test = scaler_tar.inverse_transform(y_pred_test)

y_val_multi = scaler_tar.inverse_transform(y_val_multi)
y_train_multi = scaler_tar.inverse_transform(y_train_multi)

reduce_mean_test = basic_loss_function(y_val_multi, y_pred_test)
reduce_mean_train = basic_loss_function(y_train_multi, y_pred_train)

print(f'error train: {reduce_mean_train}; error test: {reduce_mean_test}')









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



