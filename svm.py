from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import os.path

## Set data path
parsed_data_path = 'parsed_data/'

train_file = parsed_data_path + 'train.npz'
train_feature = parsed_data_path + 'train_features.npz'
val_feature = parsed_data_path + 'val_features.npz'
test_file = parsed_data_path + 'test.npz'

## Reading training and testing data
if os.path.isfile(train_feature):
    data = np.load(train_feature)
    X = data['features']
    data = np.load(val_feature)
    valX = data['features']
    #print(X)
    print('** Loaded training features from preloaded files **')
else:
    print('** Training features NOT found **')

if os.path.isfile(train_file):
    data = np.load(train_file)
    label = data['Y']
    val_label = data['valY']
    
    #print(label)
    print('** Loaded training labels from preloaded files **')
else:
    print('** Training data NOT found **')

## Prepare labels for training
Y = np.argmax(label, axis=1)
valY = np.argmax(val_label, axis=1)
#print(Y)

## Train a SVM multi-classifier
clf = svm.LinearSVC(multi_class='crammer_singer', verbose=True)
#clf = svm.SVC(verbose=True)
print(clf)
clf.fit(X, Y)
pred = clf.predict(valX)
gt = valY
print(pred)
#print(accuracy_score(gt, pred))
print(clf.score(valX, valY))
