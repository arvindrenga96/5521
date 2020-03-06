import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import q4functions
import functions
# Importing and processing digits dataset

# (b) loading digits dataset
digits = datasets.load_digits()
X = digits.data
Y = digits.target


#Q4 i

X1 = q4functions.rand_proj(X,32)
X2 = q4functions.quad_proj(X)

#calling for (i)
print("LinearSVC with X1")
functions.my_cross_val(LinearSVC, X1, Y, folds=10)
print("\nLinearSVC with X2")
functions.my_cross_val(LinearSVC,X2, Y, folds=10)

print("\nSVC with X1")
functions.my_cross_val(SVC, X1, Y, folds=10)
print("\nSVC with X2")
functions.my_cross_val(SVC, X2, Y, folds=10)

print("\nLogisticRegression with X1")
functions.my_cross_val(LogisticRegression, X1, Y, folds=10)
print("\nLogisticRegression with X2")
functions.my_cross_val(LogisticRegression, X2, Y, folds=10)

