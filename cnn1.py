from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum, Adam, SGD
import numpy as np
from tflearn.layers.normalization import batch_normalization
from data_load import read_training, read_testing
import os.path

#### PARAMs to change on differnet run/ machine ####
log_name = 'compare_results_500e'
log_dir  = '/home/abhven/ML_proj/log'

# items = ['toilet', 'bed', 'airplane', 'bench', 'guitar', 'keyboard'] 
items = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'] 

parsed_data_path = 'parsed_data/'

#####################################################

## Reading training and testing data
train_file = parsed_data_path +'train.npz'
test_file  = parsed_data_path +'test.npz'


if os.path.isfile(train_file):
	data = np.load(train_file)
	X = data['X']
	Y = data['Y']
	valX = data['valX']
	valY = data['valY']
	print('** Loaded training data from preloaded files **')
else:
	[X, Y, valX, valY ] = read_training(items)
	np.savez_compressed(train_file , X=X, Y=Y, valX=valX, valY=valY)
	print('** Parsing training data and saving to disk **')

if os.path.isfile(test_file):
	data = np.load(test_file)
	testX = data['testX']
	testY = data['testY']
	print('** Loaded testing data from preloaded files **')
else:
	[testX, testY] = read_testing(items)
	np.savez_compressed(test_file , testX=testX, testY=testY)
	print('** Parsing training data and saving to disk **')


#####################################################


#building network
network = input_data(shape=[None, 30, 30, 30, 1], name='input')
network = conv_3d(network, 32, 5, 2, padding='same', activation='leaky_relu')

network = batch_normalization(network)

#network = leaky_relu(network, alpha = 0.1)
network = dropout(network, 0.8) #dropout factor not mentioned
network = conv_3d(network, 32, 3, 1, padding='same', activation='leaky_relu')
#network = tflearn.activations.leaky_relu(network, alpha=0.1)
network = batch_normalization(network)

#network = leaky_relu(network, alpha = 0.1)
network = dropout(network, 0.8)
network = max_pool_3d(network, 2)
# network = dropout(network, 0.8)
network = batch_normalization(network)

weight = tflearn.initializations.truncated_normal (mean=0.0, stddev=0.01, seed=None)
network = fully_connected(network, 128, activation='relu', weights_init=weight)
network = dropout(network, 0.5)
network = fully_connected(network, len(items), activation='softmax', weights_init=weight)
# network = dropout(network, 0.5)

#sgd = SGD(learning_rate=0.01)
#adam = Adam(learning_rate=0.001, beta1=0.99)
momentum = Momentum(learning_rate=0.001, momentum=0.9, lr_decay=0.1, decay_step=40000) ##learning_rate=0.1, decay_step=8000 for Lidar
# reg = tflearn.regression(network, optimizer=momentum, loss='categorical_crossentropy')
network = tflearn.regression(network, optimizer=momentum, loss='categorical_crossentropy')


model = tflearn.DNN(network, tensorboard_verbose=0, 
					tensorboard_dir = log_dir, #checkpoint_path='/home/abhven/ML_proj/saved_models', 
					# best_checkpoint_path='/home/abhven/ML_proj/best_model',
					# best_val_accuracy=0.0
					)

model.fit(X, Y, n_epoch=500,  validation_set=(valX , valY), 
					shuffle=True,
          			show_metric=True, batch_size=32,
					 run_id=log_name)

score = model.evaluate(testX, testY)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
