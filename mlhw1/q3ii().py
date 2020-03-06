from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import function2

import pandas as pd

from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
digits = load_digits()

boston = load_boston()

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names

bos['Median'] = boston.target

med = bos.loc[:,'Median'].median()

bos['Median'].values[bos['Median'].values >= med] = 1
bos['Median'].values[bos['Median'].values != 1] = 0

bos=bos.rename(columns = {'Median':'y'})
bostarget = pd.DataFrame(bos['y'])
del bos['y']
bos.to_excel("boston50input.xlsx")
bostarget.to_excel("boston50output.xlsx")


boston50 = pd.read_excel('boston50input.xlsx')


boston = load_boston()
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names

bos['Median'] = boston.target

percentile75 = bos.loc[:,'Median'].quantile(.75)


bos['Median'].values[bos['Median'].values >= percentile75] = 1
bos['Median'].values[bos['Median'].values != 1] = 0

bos=bos.rename(columns = {'Median':'y'})
bostarget = pd.DataFrame(bos['y'])
del bos['y']
bos.to_excel("boston75input.xlsx")
bostarget.to_excel("boston75output.xlsx")


# test cross validation split

boston50target = pd.read_excel('boston50output.xlsx')
boston50target=boston50target.as_matrix()

boston50input = pd.read_excel('boston50input.xlsx')
boston50input=boston50input.as_matrix()

boston75target = pd.read_excel('boston75output.xlsx')
boston75target=boston75target.as_matrix()

boston75input = pd.read_excel('boston75input.xlsx')
boston75input=boston75input.as_matrix()


digits = load_digits()

k=10
print("\nError rates for LinearSVC with Boston50:")
function2.my_train_test(LinearSVC, boston50input, boston50target, 0.75, k)
k=10
print("\nError rates for LinearSVC with Boston75:")
function2.my_train_test(LinearSVC,boston75input,boston75target,0.75,k)
print("\nError rates for LinearSVC with Digits:")
function2.my_train_test(LinearSVC,digits.data,digits.target,0.75,k)

print("\nError rates for SVC with Boston50:")
function2.my_train_test(SVC,boston50input,boston50target,0.75,k)
print("\nError rates for SVC with Boston75:")
function2.my_train_test(SVC,boston75input,boston75target,0.75,k)
print("\nError rates for SVC with Digits:")
function2.my_train_test(SVC,digits.data,digits.target, 0.75,k)

print("\nError rates for LogisticRegression with Boston50:")
function2.my_train_test(LogisticRegression,boston50input,boston50target,0.75,k)
print("\nError rates for LogisticRegression with Boston75:")
function2.my_train_test(LogisticRegression,boston75input,boston75target,0.75,k)
print("\nError rates for LogisticRegression with Digits:")
function2.my_train_test(LogisticRegression,digits.data,digits.target, 0.75,k)




