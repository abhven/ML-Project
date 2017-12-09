from os import listdir
from os.path import isfile, join
from scipy.io import loadmat 
import numpy as np

def read_training( items = ['toilet', 'bed', 'airplane', 'bench', 'guitar', 'keyboard'] ):
	train_X = [] # occupancy data
	train_Y = [] #labels
	val_X = [] # occupancy data
	val_Y = [] #labels

	for index, item in enumerate(items):

		item_path = 'author_code/voxnet/3DShapeNets/volumetric_data/' + item + '/30/train/'
		item_files = [f for f in listdir(item_path) if isfile(join(item_path, f))]
		label = np.zeros(len(items), dtype=np.float32) # float32 one hot vector
		label [ index ] = 1.0 # creating one hot vector

		len_train = len( item_files)* 9 / 10 #assuming that the int is forcefully converted. This might not work on all python versions.
		len_val  = len( item_files) - len_train
		item_X = np.empty([ len_train, 30, 30, 30, 1]) # occupancy data
		item_Y = np.empty([ len_train, len(items)]) #labels
		item_X_val = np.empty([ len_val , 30, 30, 30, 1]) # occupancy data
		item_Y_val = np.empty([ len_val , len(items)]) #labels

		# Reading all the files for a specific category
		for i, inst in enumerate( item_files ):
			dat = loadmat( item_path + inst)
			voxels = np.array(dat['instance'], dtype=np.float32)
			if i < len_train: 
				item_X [ i ] = voxels[:,:,:,np.newaxis]
				item_Y [ i ] = label # one hot vector
			else:
				item_X_val [ i - len_train ] = voxels[:,:,:,np.newaxis]
				item_Y_val [ i - len_train ] = label # one hot vector 
		try:
			train_X = np.concatenate ([train_X, item_X])			
		except:
			train_X = item_X

		try:
			train_Y = np.concatenate ([train_Y, item_Y])			
		except:
			train_Y = item_Y
		
		try:
			val_X = np.concatenate ([val_X, item_X_val])			
		except:
			val_X = item_X_val
		
		try:
			val_Y = np.concatenate ([val_Y, item_Y_val])			
		except:
			val_Y = item_Y_val
	
	# print train_X.shape
	return [train_X, train_Y, val_X, val_Y]
	

def read_testing( items = ['toilet', 'bed', 'airplane', 'bench', 'guitar', 'keyboard'] ):
	test_X = [] # occupancy data
	test_Y = [] #labels

	for index, item in enumerate(items):

		item_path = 'author_code/voxnet/3DShapeNets/volumetric_data/' + item + '/30/test/'
		item_files = [f for f in listdir(item_path) if isfile(join(item_path, f))]
		label = np.zeros(len(items), dtype=np.float32) # float32 one hot vector
		label [ index ] = 1.0 # creating one hot vector

		len_test = len( item_files) #assuming that the int is forcefully converted. This might not work on all python versions.
		item_X = np.empty([ len_test, 30, 30, 30, 1]) # occupancy data
		item_Y = np.empty([ len_test, len(items)]) #labels

		# Reading all the files for a specific category
		for i, inst in enumerate( item_files ):
			dat = loadmat( item_path + inst)
			voxels = np.array(dat['instance'], dtype=np.float32)
			item_X [ i ] = voxels[:,:,:,np.newaxis]
			item_Y [ i ] = label # one hot vector

		try:
			test_X = np.concatenate ([test_X, item_X])			
		except:
			test_X = item_X

		try:
			test_Y = np.concatenate ([test_Y, item_Y])			
		except:
			test_Y = item_Y
		
	return [test_X, test_Y]
	
def main():
	read_training()

if __name__=="__main__":
	main()

'''
	
delete this if the data reading part works fine. 


	# toilet_path='author_code/voxnet/3DShapeNets/volumetric_data/toilet/30/train/'
	# bed_path='author_code/voxnet/3DShapeNets/volumetric_data/bed/30/train/'
	# airplane_path='author_code/voxnet/3DShapeNets/volumetric_data/airplane/30/train/'


	# toilet = [f for f in listdir(toilet_path) if isfile(join(toilet_path, f))]
	# bed = [f for f in listdir(bed_path) if isfile(join(bed_path, f))]
	# airplane = [f for f in listdir(airplane_path) if isfile(join(airplane_path, f))]
	
	# total_len=len(toilet) + len (bed) + len(airplane)
	# train_X = np.empty([total_len,30,30,30,1]) # occupancy data
	# train_Y = np.empty([total_len,3]) #labels

	# for i in range(len(toilet)):
	# 	dat = loadmat(toilet_path+toilet[i])
	# 	voxels = dat['instance']
	# 	train_X[i] = voxels[:,:,:,np.newaxis]
	# 	train_Y[i] = np.array([1, 0 ,0])
	
	# offset = len(toilet)

	# for i in range(len(bed)):
	# 	dat = loadmat(bed_path + bed[i])
	# 	voxels = dat['instance']
	# 	train_X[i + offset] = voxels[:,:,:,np.newaxis]
	# 	train_Y[i + offset] = np.array([0, 1 ,0])
	
	# offset = len(toilet) + len(bed)


	# for i in range(len(airplane)):
	# 	dat = loadmat(airplane_path + airplane[i])
	# 	voxels = dat['instance']
	# 	train_X[i + offset] = voxels[:,:,:,np.newaxis]
	# 	train_Y[i + offset] = np.array([0, 0 ,1])

'''