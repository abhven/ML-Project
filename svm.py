from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np

from import_data import import_data

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
