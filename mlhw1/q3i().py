from random import seed

import numpy
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

k=10
print("\nLinearSVC with Boston50:")
functions.my_cross_val(LinearSVC,boston50input,boston50target,k)
print("\nLinearSVC with Boston75:")
functions.my_cross_val(LinearSVC,boston75input,boston75target,k)
print("\nLinearSVC with Digits:")
functions.my_cross_val(LinearSVC,digits.data,digits.target, k)

print("\nSVC with Boston50:")
functions.my_cross_val(SVC,boston50input,boston50target,k)
print("\nSVC with Boston75:")
functions.my_cross_val(SVC,boston75input,boston75target,k)
print("\nSVC with Digits:")
functions.my_cross_val(SVC,digits.data,digits.target, k)

print("\nLogisticRegression with Boston50:")
functions.my_cross_val(LogisticRegression,boston50input,boston50target,k)
print("\nLogisticRegression with Boston75:")
functions.my_cross_val(LogisticRegression,boston75input,boston75target,k)
print("\nLogisticRegression with Digits:")
functions.my_cross_val(LogisticRegression,digits.data,digits.target, k)




