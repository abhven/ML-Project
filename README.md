# ML-Project
We have implemented 3 models
## Basline model
This is the baseline model re implimented in tensorflow.
To run this, set the training mode to 1 and you can train the model.
```
python cnn_baseline.py
```
## SVM atop Basline model
This method uses the pretrained model from the baseline and uses an SVM classifier on the feature vectors generated from the baseline. 
To run this, first set the training mode to 1 in cnn_baseline.py and make sure that the model is trained
```
python cnn_baseline.py
```
Then change the training mode to 0 and run again. This will save the features as .npz file. 
```
python cnn_baseline.py
```
Then run svm.py for the classification.
```
python svm.py
```

## Deep Neural Network model
This is the 4 Conv + 2FC model in tensorflow. This performs better than the baseline version.
To run this, set the training mode to 1 and you can train the model.
```
python cnn_deepNN.py
```

## Author's code 

The authors code is implemented in Theano. This is the [link](https://github.com/dimatura/voxnet) to the orginal Voxnet implementation.




