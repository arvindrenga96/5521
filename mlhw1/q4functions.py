import numpy as np

def rand_proj(X,d):
    G = np.random.normal(0, 1, size=(64, 32))
    X1 = np.matmul(X, G)
    return X1;


def quad_proj(X):
    no_of_rows , no_of_cols= len(X), len(X[0])
    #total coloums are summation of original + original^2 + n(n-1)/2 as it is NC2 combination
    #print(no_of_rows,no_of_cols)
    total_cols_aftertransform = np.int(no_of_cols*2 + ((no_of_cols*(no_of_cols-1))/2))
    #print(total_cols_aftertransform)
    X_transformed = np.empty((no_of_rows,total_cols_aftertransform))

    for i in range(no_of_rows):
        for j in range(no_of_cols):
             X_transformed[i][j]= X[i][j]

    for i in range(no_of_rows):
        for j in range(no_of_cols,no_of_cols*2):
            X_transformed[i][j] = X[i][j-no_of_cols]*X[i][j-no_of_cols]
            #print(X_transformed[i][j])

    for i in range(no_of_rows):
        count = no_of_cols * 2
        for j in range(no_of_cols):
            for k in range(j+1, no_of_cols):
                X_transformed[i][count] = X[i][j] * X[i][k]
                count=count+1
    return X_transformed