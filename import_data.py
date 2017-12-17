import numpy as np
import os.path

def import_data(parsed_data_path):
	## Set data path
	train_feature = parsed_data_path + 'train_features.npz'
	val_feature = parsed_data_path + 'val_features.npz'
	test_feature = parsed_data_path + 'test_features.npz'
	train_file = parsed_data_path + 'train.npz'
	test_file = parsed_data_path + 'test.npz'

	## Reading and prepare data
	if os.path.isfile(train_feature):
	    data = np.load(train_feature)
	    X = data['features']
	    #print(X)
	    print('** loaded Training features from preloaded files **')
	else:
	    print('** Training features NOT found **')

	if os.path.isfile(val_feature):
	    data = np.load(val_feature)
	    valX = data['features']
	    #print(valX)
	    print('** loaded Validation features from preloaded files **')
	else:
	    print('** Validation features NOT found **')

	if os.path.isfile(test_feature):
	    data = np.load(test_feature)
	    testX = data['features']
	    #print(testX)
	    print('** loaded Test features from preloaded files **')
	else:
	    print('** Test features NOT found **')

	if os.path.isfile(train_file):
	    data = np.load(train_file)
	    label = data['Y']
	    Y = np.argmax(label, axis=1)
	    val_label = data['valY']
	    valY = np.argmax(val_label, axis=1)
	    #print(Y)
	    #print(valY)
	    print('** loaded Training labels from preloaded files **')
	else:
	    print('** Training data NOT found **')

	if os.path.isfile(test_file):
	    data = np.load(test_file)
	    test_label = data['testY']
	    testY = np.argmax(test_label, axis=1)
	    #print(testY)
	    print('** loaded Test labels from preloaded files **')
	else:
	    print('** Test data NOT found **')

	return X, Y, valX, valY, testX, testY
