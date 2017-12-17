import itertools
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, make_scorer
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



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
best_nonlinear_clf = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], verbose=True)
print(best_nonlinear_clf)
best_nonlinear_clf.fit(X, Y)
print(best_nonlinear_clf.score(testX, testY))
predY = best_nonlinear_clf.predict(testX)


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
                      title='Normalized confusion matrix')

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



