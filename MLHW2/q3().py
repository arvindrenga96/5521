from random import seed
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import ParametricClassfier
import Mycrossval
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
seed(1)
boston50target = pd.read_excel('boston50output.xlsx')
boston50target=boston50target.as_matrix()

boston50input = pd.read_excel('boston50input.xlsx')
boston50input=boston50input.as_matrix()

boston75target = pd.read_excel('boston75output.xlsx')
boston75target=boston75target.as_matrix()

boston75input = pd.read_excel('boston75input.xlsx')
boston75input=boston75input.as_matrix()


digits = load_digits()


#train model

k=10
print("\nMultiGaussClassify with full covariance matrix on Boston50")
Mycrossval.my_cross_val(ParametricClassfier.MultiGaussClassify,boston50input,boston50target,k,diag =False,K=2)
print("\nMultiGaussClassify with full covariance matrix on Boston75")
Mycrossval.my_cross_val(ParametricClassfier.MultiGaussClassify,boston75input,boston75target,k,diag =False,K=2)
print("\nMultiGaussClassify with full covariance matrix on Digits")
Mycrossval.my_cross_val(ParametricClassfier.MultiGaussClassify,digits.data,digits.target,k,diag =False,K=10)

print("\nMultiGaussClassify with Diagonal covariance matrix on Boston50")
Mycrossval.my_cross_val(ParametricClassfier.MultiGaussClassify,boston50input,boston50target,k,diag =True,K=2)
print("\nMultiGaussClassify with Diagonal covariance matrix on Boston75")
Mycrossval.my_cross_val(ParametricClassfier.MultiGaussClassify,boston75input,boston75target,k,diag =True,K=2)
print("\nMultiGaussClassify with Diagonal covariance matrix on Digits")
Mycrossval.my_cross_val(ParametricClassfier.MultiGaussClassify,digits.data,digits.target,k,diag =True,K=10)

print("\nLogisticRegression with Boston50:")
Mycrossval.my_cross_val(LogisticRegression,boston50input,boston50target,k)
print("\nLogisticRegression with Boston75:")
Mycrossval.my_cross_val(LogisticRegression,boston75input,boston75target,k)
print("\nLogisticRegression with Digits:")
Mycrossval.my_cross_val(LogisticRegression,digits.data,digits.target, k)
