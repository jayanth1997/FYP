# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:49:47 2019

@author: Jayanth
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:52:10 2019

@author: Jayanth
"""

# This script defines a GRU-RNN to map the cepstral components of the signal

# This import makes Python use 'print' as in Python 3.x

from __future__ import print_function
# importing the required module 
import matplotlib.pyplot as plt 
import os
import h5py
import numpy as np
from keras.layers import Dense, Dropout, GRU, CuDNNGRU,CuDNNLSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop,SGD,Adam,Adagrad
from tfglib import construct_table as ct, utils

#######################
# Sizes and constants #
#######################
# Batch shape
batch_size = 408
tsteps = 50
data_dim = 40

# Other constants
epochs = 400
# epochs = 25

#############
# Load data #
#############
#  Switch to decide if datatable must be build or can be loaded from a file
build_datatable = False

print('Starting...')

a=np.loadtxt('SF1_mfcc.txt')
e=int(len(a)/40)
s1_train_data=np.reshape(a,(e,40))
b=np.loadtxt('TM1_mfcc.txt')
e=int(len(b)/40)
t1_train_data=np.reshape(b,(e,40))
c=np.loadtxt('SF1_val_mfcc.txt')
e=int(len(c)/40)
s1_valid_data=np.reshape(c,(e,40))
d=np.loadtxt('TM1_val_mfcc.txt')
e=int(len(d)/40)
t1_valid_data=np.reshape(d,(e,40))
g=np.loadtxt('SF1_test_mfcc.txt')
e=int(len(g)/40)
s1_test_data=np.reshape(g,(e,40))
h=np.loadtxt('TM1_test_mfcc.txt')
e=int(len(h)/40)
t1_test_data=np.reshape(h,(e,40))

s1_train_data=s1_train_data[0:204000,:]
t1_train_data=t1_train_data[0:204000,:]
s1_valid_data=s1_valid_data[0:20400,:]
t1_valid_data=t1_valid_data[0:20400,:]


################
# Prepare data #
################
# Take MCP parameter columns
src_train_data = s1_train_data  # Source data
trg_train_data = t1_train_data  # Target data

src_valid_data = s1_valid_data # Source data
trg_valid_data = t1_valid_data # Target data

src_test_data = s1_test_data  # Source data
trg_test_data = t1_test_data  # Target data



# Remove means and normalize
src_train_mean = np.mean(src_train_data, axis=0) #axis0 =column
src_train_std = np.std(src_train_data, axis=0)

src_train_data = (src_train_data - src_train_mean) / src_train_std
src_valid_data = (src_valid_data - src_train_mean) / src_train_std
src_test_data = (src_test_data - src_train_mean) / src_train_std

trg_train_mean = np.mean(trg_train_data, axis=0)
trg_train_std = np.std(trg_train_data, axis=0)

trg_train_data = (trg_train_data - trg_train_mean) / trg_train_std
trg_valid_data = (trg_valid_data - trg_train_mean) / trg_train_std

# Zero-pad and reshape data
src_train_data = utils.reshape_lstm1(src_train_data, tsteps, data_dim)
src_valid_data = utils.reshape_lstm1(src_valid_data, tsteps, data_dim)
src_test_data = utils.reshape_lstm1(src_test_data, tsteps, data_dim)

trg_train_data = utils.reshape_lstm1(trg_train_data, tsteps, data_dim)
trg_valid_data = utils.reshape_lstm1(trg_valid_data, tsteps, data_dim)

# Save training statistics
with h5py.File('Intermediate_results/mcp_train_stats.h5', 'w') as f:
  h5_src_train_mean = f.create_dataset("src_train_mean", data=src_train_mean)
  h5_src_train_std = f.create_dataset("src_train_std", data=src_train_std)
  h5_trg_train_mean = f.create_dataset("trg_train_mean", data=trg_train_mean)
  h5_trg_train_std = f.create_dataset("trg_train_std", data=trg_train_std)

  f.close()
  
  

################
# Define Model #
################
# Define an GRU-based RNN
print('Creating Model')
'''model = Sequential()

model.add(CuDNNGRU(units=70,
              batch_input_shape=(batch_size, tsteps, data_dim),
              return_sequences=True,
              stateful=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(data_dim)))

rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop, metrics=['accuracy'])'''
model = Sequential()
model.add(CuDNNGR(units=100,
              batch_input_shape=(batch_size, tsteps, data_dim),
              return_sequences=True,
              stateful=True))
model.add(CuDNNGRU(100, return_sequences=True,
              stateful=True ))
model.add(CuDNNGRU(100, return_sequences=True,
              stateful=True ))
model.add(CuDNNGRU(100, return_sequences=True,
              stateful=True ))
model.add(CuDNNGRU(100, return_sequences=True,
              stateful=True ))
model.add(TimeDistributed(Dense(data_dim)))

rmsprop = SGD(lr=0.001,momentum=0.9)
model.compile(loss='mse', optimizer=rmsprop, metrics=['accuracy'])



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
model.save_weights('Intermediate_results/mcp_weights.h5')

with open('Intermediate_results/mcp_model.json', 'w') as model_json:
  model_json.write(model.to_json())

print('Saving training results')
with h5py.File(os.path.join('training_results', 'baseline', 'mcp_history.h5'),
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


  

