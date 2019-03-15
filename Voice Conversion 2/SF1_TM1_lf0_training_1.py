# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:58:02 2019

@author: Jayanth
"""

# This script initializes and trains an LSTM-based RNN for log(f0) mapping

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import h5py
import numpy as np

from keras.layers import Dense, LSTM,CuDNNGRU,CuDNNLSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop
from tfglib import construct_table as ct, utils


#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 8
tsteps = 50
data_dim = 1

# Other constants
epochs = 50

#############
# Load data #
#############
# Switch to decide if datatable must be build or can be loaded from a file
build_datatable = False

print('Starting...')

#Read lf0 values
s1_train_data=np.loadtxt('SF1_lf0.txt')
t1_train_data=np.loadtxt('TM1_lf0.txt')
s1_valid_data=np.loadtxt('SF1_val_lf0.txt')
t1_valid_data=np.loadtxt('TM1_val_lf0.txt')
s1_test_data=np.loadtxt('SF1_test_lf0.txt')
t1_test_data=np.loadtxt('TM1_test_lf0.txt')

s1_train_data=s1_train_data[0:204000]
t1_train_data=t1_train_data[0:204000]
s1_valid_data=s1_valid_data[0:20400]
t1_valid_data=t1_valid_data[0:20400]
################
# Prepare data #
################
# Number of training samples
nb_samples = 204000
# Take lfo columns
src_train_data = s1_train_data
trg_train_data = t1_train_data
src_valid_data = s1_valid_data    
trg_valid_data = t1_valid_data    
src_test_data = s1_test_data
trg_test_data = t1_test_data


# Remove means and normalize
src_train_mean = np.mean(src_train_data,axis=0)
src_train_std = np.std(src_train_data, axis=0)

src_train_data = (src_train_data - src_train_mean) / src_train_std
src_valid_data = (src_valid_data - src_train_mean) / src_train_std
src_test_data = (src_test_data - src_train_mean) / src_train_std

trg_train_mean = np.mean(trg_train_data, axis=0)
trg_train_std = np.std(trg_train_data, axis=0)

trg_train_data = (trg_train_data - trg_train_mean) / trg_train_std
trg_valid_data = (trg_valid_data - trg_train_mean) / trg_train_std
# trg_test_data[:, 0] = (trg_test_data[:, 0] - trg_train_mean) / trg_train_std

# Zero-pad and reshape data
src_train_data = utils.reshape_lstm(src_train_data, tsteps, data_dim)
src_valid_data = utils.reshape_lstm(src_valid_data, tsteps, data_dim)
src_test_data = utils.reshape_lstm(src_test_data, tsteps, data_dim)
trg_train_data = utils.reshape_lstm(trg_train_data, tsteps, data_dim)
trg_valid_data = utils.reshape_lstm(trg_valid_data, tsteps, data_dim)
trg_test_data = utils.reshape_lstm(trg_test_data, tsteps, data_dim)


# Save training statistics
with h5py.File('Intermediate_results/lf0_train_stats.h5', 'w') as f:
  h5_src_train_mean = f.create_dataset("src_train_mean", data=src_train_mean)
  h5_src_train_std = f.create_dataset("src_train_std", data=src_train_std)
  h5_trg_train_mean = f.create_dataset("trg_train_mean", data=trg_train_mean)
  h5_trg_train_std = f.create_dataset("trg_train_std", data=trg_train_std)

  f.close()

################
# Define Model #
################
# Define an LSTM-based RNN
print('Creating Model')
model = Sequential()

model.add(CuDNNLSTM(units=100,
               batch_input_shape=(batch_size, tsteps, data_dim),
               return_sequences=True,
               stateful=True))
model.add(TimeDistributed(Dense(1)))

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop, metrics=["accuracy"])

###############
# Train model #
###############
print('Training')
epoch = list(range(epochs))
loss = []
val_loss = []

for i in range(epochs):
  print('Epoch', i, '/', epochs)
  history = model.fit(src_train_data,
                      trg_train_data,
                      batch_size=batch_size,
                      verbose=1,
                      epochs=1,
                      shuffle=False,
                      validation_data=(src_valid_data, trg_valid_data))

  loss.append(history.history['loss'])
  val_loss.append(history.history['val_loss'])

  model.reset_states()

print('Saving model')
model.save_weights('Intermediate_results/lf0_weights.h5')

with open('Intermediate_results/lf0_model.json', 'w') as model_json:
  model_json.write(model.to_json())

print('Saving training results')
with h5py.File(os.path.join('training_results', 'baseline', 'lf0_history.h5'),
               'w') as hist_file:
  hist_file.create_dataset('loss', data=loss,
                           compression='gzip', compression_opts=9)
  hist_file.create_dataset('val_loss', data=val_loss,
                           compression='gzip', compression_opts=9)
  hist_file.create_dataset('epoch', data=epoch, compression='gzip',
                           compression_opts=9)

  hist_file.close()

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')


