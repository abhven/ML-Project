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
log_name = 'deep_train'
log_dir  = '/home/abhven/ML_proj/log'
model_file = 'saved_models/' +'deep_model' +'.tf1'

# items = ['toilet', 'bed', 'airplane', 'bench', 'guitar', 'keyboard'] 
items = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'] 

parsed_data_path = 'parsed_data/'

train_mode = 0
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

network = dropout(network, 0.8) #dropout factor not mentioned
network = conv_3d(network, 32, 3, 1, padding='same', activation='leaky_relu')
network = batch_normalization(network)

network = dropout(network, 0.8)
network = max_pool_3d(network, 2)
network = batch_normalization(network)

network = dropout(network, 0.8) #dropout factor not mentioned
network = conv_3d(network, 32, 3, 1, padding='same', activation='leaky_relu')
network = batch_normalization(network)

network = dropout(network, 0.8) #dropout factor not mentioned
network = conv_3d(network, 32, 3, 1, padding='same', activation='leaky_relu')
network = batch_normalization(network)

weight = tflearn.initializations.truncated_normal (mean=0.0, stddev=0.01, seed=None)
network = fully_connected(network, 128, activation='relu', weights_init=weight)
base_network = dropout(network, 0.5)
base_network = fully_connected(base_network, len(items), activation='softmax', weights_init=weight)
# network = dropout(network, 0.5)

#sgd = SGD(learning_rate=0.01)
#adam = Adam(learning_rate=0.001, beta1=0.99)
momentum = Momentum(learning_rate=0.001, momentum=0.9, lr_decay=0.1, decay_step=40000) ##learning_rate=0.1, decay_step=8000 for Lidar
# reg = tflearn.regression(network, optimizer=momentum, loss='categorical_crossentropy')
base_network = tflearn.regression(base_network, optimizer=momentum, loss='categorical_crossentropy')


model = tflearn.DNN(base_network, tensorboard_verbose=0, 
					tensorboard_dir = log_dir, #checkpoint_path='/home/abhven/ML_proj/saved_models', 
					# best_checkpoint_path='/home/abhven/ML_proj/best_model',
					# best_val_accuracy=0.0
					)

if train_mode:
    model.fit(X, Y, n_epoch=100,  validation_set=(valX , valY), 
					shuffle=True,
          			show_metric=True, batch_size=32,
					 run_id=log_name)
    model.save(model_file)
else:
    model.load(model_file)
    # new_network = fully_connected(network, 16, activation='relu', weights_init=weight)
    new_model = tflearn.DNN(network, tensorboard_verbose=0, 
					tensorboard_dir = log_dir)
    ## Training features
    features=np.empty([len(X), 128])
    for i in range(int(len(X)/100)):
    	features[(i)*100 : (i+1)*100]= new_model.predict(X[(i)*100 : (i+1)*100])

	features[(int(len(X)/100))*100 : len(X)]= new_model.predict(X[(int(len(X)/100))*100 : len(X)])
	np.savez_compressed('deep_train_features.npz', features=features)

	## Validation Features
	val_features=np.empty([len(valX), 128])
    for i in range(int(len(valX)/100)):
    	val_features[(i)*100 : (i+1)*100]= new_model.predict(valX[(i)*100 : (i+1)*100])

	val_features[(int(len(valX)/100))*100 : len(valX)]= new_model.predict(valX[(int(len(valX)/100))*100 : len(valX)])
	np.savez_compressed('deep_val_features.npz', features=val_features)

	## Validation Features
	test_features=np.empty([len(testX), 128])
    for i in range(int(len(testX)/100)):
    	test_features[(i)*100 : (i+1)*100]= new_model.predict(testX[(i)*100 : (i+1)*100])

	test_features[(int(len(testX)/100))*100 : len(testX)]= new_model.predict(testX[(int(len(testX)/100))*100 : len(testX)])
	np.savez_compressed('deep_test_features.npz', features=test_features)


score = model.evaluate(testX, testY)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
