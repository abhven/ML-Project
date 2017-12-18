from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import itertools
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
import numpy as np

from import_data import import_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


#iris = datasets.load_iris()
#class_names = iris.target_names
#print(class_names)
class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=20)
    plt.yticks(tick_marks, classes, size=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=20)

    plt.tight_layout()
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)



# Load data
parsed_data_path = 'parsed_data/'
[X, Y, valX, valY, testX, testY] = import_data(parsed_data_path)

## Cross Validation
accuracy_scorer = make_scorer(accuracy_score)

## CV for Nonlinear kernel SVM
tuned_parameters = [{'gamma': [0.3, 0.2, 0.1], 'C': [100, 500, 1000]}]
clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring=accuracy_scorer, n_jobs=12, verbose=1, return_train_score=True)
clf.fit(valX, valY)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

## Re-fit a nonlinear SVM using the best params
best_params = clf.best_params_
best_nonlinear_clf = svm.SVC(probability=True, kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], verbose=True)
print(best_nonlinear_clf)
best_nonlinear_clf.fit(X, Y)
print(best_nonlinear_clf.score(testX, testY))
#predY = best_nonlinear_clf.predict(testX)
probY = best_nonlinear_clf.predict_proba(testX)

#skplt.metrics.plot_roc_curve(testY, probY, classes=class_names)
#plt.show()


y_test = label_binarize(testY, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Compute ROC curve and ROC area for each class
n_classes= 10
lw = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probY[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probY.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size=20)
plt.ylabel('True Positive Rate', size=20)
plt.title('Multi-class Receiver Operating Characteristic (ROC) Curves', size=20)
plt.legend(loc="lower right")
plt.show()




# Compute confusion matrix
cnf_matrix = confusion_matrix(testY, predY)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized Confusion Matrix')

plt.show()




## CV for Linear kernel SVM
tuned_parameters = [{'C' : [0.01, 0.1, 0.2, 0.3, 0.4]}]
clf = GridSearchCV(svm.LinearSVC(multi_class="crammer_singer"), tuned_parameters, scoring=accuracy_scorer, n_jobs=12, verbose=1, return_train_score=True)
clf.fit(valX, valY)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

## Re-fit a linear SVM using the best params
best_params = clf.best_params_
best_linear_clf = svm.LinearSVC(multi_class="crammer_singer", C=best_params['C'], max_iter=3000, verbose=True)
print(best_linear_clf)
best_linear_clf.fit(X, Y)
print(best_linear_clf.score(testX, testY))



