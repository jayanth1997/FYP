# This is a script for initializing and training a fully-connected DNN

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function
import os
import h5py
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import RMSprop
from tfglib import construct_table as ct
from tfglib.utils import apply_context

#############
# Load data #
#############
#  Switch to decide if datatable must be build or can be loaded from a file
build_datatable = False

print('Starting...')
s1_train_data=np.loadtxt('SF1_mvf.txt')
t1_train_data=np.loadtxt('TM1_mvf.txt')
s1_valid_data=np.loadtxt('SF1_val_mvf.txt')
t1_valid_data=np.loadtxt('TM1_val_mvf.txt')
s1_test_data=np.loadtxt('SF1_test_mvf.txt')
t1_test_data=np.loadtxt('TM1_test_mvf.txt')

s1_train_data=s1_train_data[0:204000]
t1_train_data=t1_train_data[0:204000]
s1_valid_data=s1_valid_data[0:20400]
t1_valid_data=t1_valid_data[0:20400]



#######################
# Sizes and constants #
#######################
batch_size = 300
nb_epochs = 700
learning_rate = 0.00000055
context_size = 1

################
# Prepare data #
################
# Randomize frames
# np.random.shuffle(train_data)


# Split into train and validation (17500 train, 2500 validation)
src_train_frames = s1_train_data
trg_train_frames = t1_train_data


src_valid_frames = s1_valid_data
trg_valid_frames = t1_valid_data # Target data

# Normalize data
src_train_mean = np.mean(src_train_frames, axis=0)
src_train_std = np.std(src_train_frames, axis=0)
trg_train_mean = np.mean(trg_train_frames, axis=0)
trg_train_std = np.std(trg_train_frames, axis=0)

src_train_frames = (src_train_frames - src_train_mean) / src_train_std
src_valid_frames = (src_valid_frames - src_train_mean) / src_train_std

trg_train_frames = (trg_train_frames - trg_train_mean) / trg_train_std
trg_valid_frames = (trg_valid_frames - trg_train_mean) / trg_train_std


# Save training statistics
with h5py.File('Intermediate_results/mvf_train_stats.h5', 'w') as f:
  h5_src_train_mean = f.create_dataset("src_train_mean", data=src_train_mean)
  h5_src_train_std = f.create_dataset("src_train_std", data=src_train_std)
  h5_trg_train_mean = f.create_dataset("trg_train_mean", data=trg_train_mean)
  h5_trg_train_std = f.create_dataset("trg_train_std", data=trg_train_std)

  f.close()

# Apply context
src_train_frames_context = src_train_frames
src_valid_frames_context = src_valid_frames


# exit()

################
# Define Model #
################
# Adjust DNN sizes to implement context
print('Evaluate DNN...')
model = Sequential()

# model.add(Dense(100, input_dim=2))
model.add(Dense(100, input_dim=1))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))

model.add(Dense(1, activation='linear'))

# sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, clipnorm=10)
rmsprop = RMSprop(lr=learning_rate)

# model.compile(loss='mse', optimizer=sgd)
model.compile(loss='mae', optimizer=rmsprop)

###############
# Train model #
###############
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    mode='min')

history = model.fit(
    src_train_frames_context,
    trg_train_frames,
    batch_size=batch_size,
    nb_epoch=nb_epochs,
    verbose=1,
    validation_data=(src_valid_frames_context, trg_valid_frames),
    callbacks=[reduce_lr]
    )

print('Saving model')
model.save_weights('Intermediate_results/mvf_weights.h5')

with open('Intermediate_results/mvf_model.json', 'w') as model_json:
  model_json.write(model.to_json())

print('Saving training results')
with h5py.File(os.path.join('training_results', 'baseline', 'mvf_history.h5'),
               'w') as hist_file:
  hist_file.create_dataset('loss', data=history.history['loss'],
                           compression='gzip', compression_opts=9)
  hist_file.create_dataset('val_loss', data=history.history['val_loss'],
                           compression='gzip', compression_opts=9)
  hist_file.create_dataset('epoch', data=history.epoch, compression='gzip',
                           compression_opts=9)

  hist_file.close()

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')



