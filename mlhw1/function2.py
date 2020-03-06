import numpy as np
import sklearn
import pandas as pd
from random import randrange
from sklearn.datasets import load_digits
from random import random
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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


def train_test_split(dataset,target, split=0.60):
    trainx = list()
    trainy = list()
    train_size = split * len(dataset)
    dataset_copy = dataset
    target_copy = target
    while len(trainx) < train_size:
        index = randrange(len(dataset_copy))
        last= dataset_copy[index];
        np.delete(dataset_copy, index)
        lasttarget = target_copy[index]
        np.delete(target_copy,index)
        trainx.append(last)
        trainy.append(lasttarget)
    return trainx, trainy,dataset_copy,target_copy


def my_train_test(method, matrix, target, pie=0.75, folds=10):
    res = list()
    for i in range(folds):
        Xtrain, ytrain,Xtest,ytest = train_test_split(matrix, target, pie)
        res.append(cross_val(method, Xtrain, ytrain, Xtest, ytest))
        print("FOLD", i + 1, ":", res[i])
    Mean = sum(res) / len(res)
    print("MEAN", Mean)
    StdDev = stddev(res)
    print("STDDEV", stddev(res))


def cross_val(method, X, y, testX, testY):
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

    return (1 - count / length)


