import numpy as np
import sklearn
import pandas as pd
from random import seed
from random import randrange
from sklearn.datasets import load_digits

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import ParametricClassfier

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data) / float(n)  # in Python 2 use sum(data)/float(n)


def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x - c) ** 2 for x in data)
    return ss


def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss / (n - ddof)
    return pvar ** 0.5


def my_cross_val(method: object, matrix: object, target: object, folds: object = 5, diag: object = False, K: object = 2) -> object:
    matrix_split = list()
    matrix_split_results = list()
    matrix_copy = matrix.copy()
    matrix_target_copy = target.copy()
    fold_size = int(len(matrix) / folds)
    rem = int(len(matrix)) % folds
    for i in range(folds):
        fold = list()
        foldtarget = list()
        if rem >= 1:
            while len(fold) < fold_size + 1:
                index = randrange(len(matrix_copy))
                last= matrix_copy[index]
                np.delete(matrix_copy, index)
                lasttarget = matrix_target_copy[index]
                np.delete(matrix_target_copy, index)
                foldtarget.append(lasttarget)
                fold.append(last)
            matrix_split.append(fold)
            matrix_split_results.append(foldtarget)
            rem = rem - 1;
        else:
            while len(fold) < fold_size:
                index = randrange(len(matrix_copy))
                last= matrix_copy[index]
                np.delete(matrix_copy, index)
                lasttarget = matrix_target_copy[index]
                np.delete(matrix_target_copy, index)
                foldtarget.append(lasttarget)
                fold.append(last)
            matrix_split.append(fold)
            matrix_split_results.append(foldtarget)
    X, y = matrix_split, matrix_split_results
    res = list()
    for i in range(folds):
        trainx = X.copy()
        train = y.copy()
        testx = X[i]
        test = y[i]
        del train[i]
        del trainx[i]
        train = np.concatenate(train)
        trainx = np.concatenate(trainx)
        res.append(cross_val(method, trainx, train, testx, test,diag,K))
    for i in range(folds):
        print("FOLD",i+1,":",res[i])
    Mean = sum(res) / len(res)
    print("MEAN", Mean)
    StdDev = stddev(res)
    print("STDDEV", stddev(res))


def cross_val(method, X, y, testX, testY,diagonal=False,K=2):
    if method == LinearSVC:
        clf = LinearSVC(random_state=0, tol=1e-5)
        # print(X)
        # print(y[1])
        clf.fit(X, y)
        ypredict = clf.predict(testX)
        length = len(testX)
        count = 0
        for i in range(length):
            if ypredict[i] == testY[i]:
                count = count + 1

    elif method == SVC:
        clf = SVC()
        clf.fit(X, y)
        ypredict = clf.predict(testX)
        length = len(testX)
        count = 0
        for i in range(length):
            if ypredict[i] == testY[i]:
                count = count + 1

    elif method == LogisticRegression:
        clf = LogisticRegression()
        clf.fit(X, y)
        ypredict = clf.predict(testX)
        length = len(testX)
        count = 0
        for i in range(length):
            if ypredict[i] == testY[i]:
                count = count + 1
    elif method == ParametricClassfier.MultiGaussClassify:
        clf = ParametricClassfier.MultiGaussClassify(K)
        clf.fit(X, y, Diagonal)
        ypredict = clf.predict(np.asarray(testX))
        length = len(testX)
        count = 0
        for i in range(length):
            if ypredict[i] == testY[i]:
                count = count + 1
    return (1 - count / length)