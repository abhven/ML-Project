import timeit
from skrvm import RVC
import numpy as np
import os.path

from import_data import import_data

## Set data path
parsed_data_path = 'parsed_data/'
[X, Y, valX, valY, testX, testY] = import_data(parsed_data_path)

## Train a RVM
clf = RVC(verbose=True)
print(clf)
clf.fit(valX, valY)
clf.score(testX, testY)
