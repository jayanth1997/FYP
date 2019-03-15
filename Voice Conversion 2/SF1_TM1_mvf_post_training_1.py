# Created by albert aparicio on 28/10/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os

import h5py
import matplotlib
import numpy as np
from ahoproc_tools.error_metrics import RMSE
from keras.models import model_from_json
from keras.optimizers import RMSprop
from tfglib.utils import apply_context

matplotlib.use('TKagg')
from matplotlib import pyplot as plt

#######################
# Sizes and constants #
#######################
batch_size = 300
nb_epochs = 700
learning_rate = 0.00000055
context_size = 1

##############
# Load model #
##############
print('Loading model...', end='')
with open('Intermediate_results/mvf_model.json', 'r') as model_json:
  model = model_from_json(model_json.read())

model.load_weights('Intermediate_results/mvf_weights.h5')

rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mae', optimizer=rmsprop)

#############
# Load data #
#############
# Load training statistics
with h5py.File('Intermediate_results/mvf_train_stats.h5', 'r') as train_stats:
  src_train_mean = train_stats['src_train_mean'].value
  src_train_std = train_stats['src_train_std'].value
  trg_train_mean = train_stats['trg_train_mean'].value
  trg_train_std = train_stats['trg_train_std'].value

  train_stats.close()

# Load test data
print('Loading test data...', end='')

s1_test_data = np.loadtxt('SF1_test_mvf.txt')

src_test_data = s1_test_data
src_test_data = (src_test_data - src_train_mean) / src_train_std

t1_test_data = np.loadtxt('TM1_test_mvf.txt')

trg_test_data = t1_test_data

# Apply context
src_test_data_context = src_test_data
print('done')


################
# Predict data #
################
print('Predicting')
prediction = model.predict(src_test_data_context)

# De-normalize predicted output
prediction = (prediction * trg_train_std) + trg_train_mean

#################
# Error metrics #
#################
# Compute and print RMSE of test data
'''rmse_test = RMSE(
    trg_test_data[:, 0],
    prediction[:, 0],
    mask=trg_test_data[:, 1]
    )

print('Test RMSE: ', rmse_test)'''


# Load training parameters and save loss curves
with h5py.File('training_results/baseline/mvf_history.h5', 'r') as hist_file:
  loss = hist_file['loss'][:]
  val_loss = hist_file['val_loss'][:]
  epoch = hist_file['epoch'][:]

  hist_file.close()

print('Saving loss curves')

plt.plot(epoch, loss, epoch, val_loss)
plt.legend(['loss', 'val_loss'], loc='best')
plt.grid(b=True)
plt.suptitle('Baseline MVF Loss curves')
plt.savefig(os.path.join('training_results', 'baseline', 'mvf_loss_curves.eps'),
            bbox_inches='tight')

# # Histogram of predicted training data and training data itself
# plt.hist(prediction[:, 0], bins=100)
# plt.title('Prediction frames')
# plt.savefig('prediction_hist.png', bbox_inches='tight')
# plt.show()

# # Histogram of training samples
# plt.figure()
# plt.hist(vf_gtruth, bins=100)
# plt.title('Training target frames')
# plt.savefig('gtruth_hist.png', bbox_inches='tight')
# plt.show()

print('========================' + '\n' +
      '======= FINISHED =======' + '\n' +
      '========================')

#np.savetxt('mvf_predicted_scientific.txt',prediction,fmt='%.1e')
np.savetxt('mvf_predicted_float.txt',prediction,fmt='%f')
