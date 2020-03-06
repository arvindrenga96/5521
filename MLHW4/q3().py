from random import seed
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import Mycrossval
import pandas as pd
import MySVM2
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



#train model
k=5

print("\nMySVM2 with Boston50:")
Mycrossval.my_cross_val("MySVM2",boston50input,boston50target,k)
print("\nMySVM2 with Boston75:")
Mycrossval.my_cross_val("MySVM2",boston75input,boston75target,k)
print("\nLogisticRegression with Boston50:")
Mycrossval.my_cross_val(LogisticRegression,boston50input,boston50target,k)
print("\nLogisticRegression with Boston75:")
Mycrossval.my_cross_val(LogisticRegression,boston75input,boston75target,k)

