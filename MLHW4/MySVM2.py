import numpy as np
import math
def sigmoid(z):
    return 1/(1+np.exp(-z))
sigmoid_vectorized = np.vectorize(sigmoid)
def sign(x):
    return 1 - (x >= 0)
class MySVM2:

    def __init__(self, d):
        self.coef = 0.02 * np.random.random_sample(
            (d)) - 0.01  # initializing w with values within range of [-0.01,0.01]
        self.mean = 0
        self.std = 1

    def fit(self, X: object, y: object,learning_factor,n_epoch) -> object:
        self.coef = self.coefficients_sgd(X,y,learning_factor,n_epoch)
        return self

    # Estimate logistic regression coefficients using stochastic gradient descent
    def coefficients_sgd(self, X,y, l_rate, n_epoch):
        num_samples = X.shape[0]
        x0 = np.ones((num_samples,))
        self.mean=np.mean(X, axis=0)
        self.std= np.std(X,axis=0)
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X_new = np.insert(X_normalized, 0, x0, axis=1)
        Coef = self.coef
        eta =l_rate;
        for epoch in range(1,n_epoch):
            for i,x in enumerate(X):
                if(y[i]*np.dot(X[i],Coef))<1:
                    Coef = Coef + eta*((X[i]*y[i])-(Coef/epoch))
                else:
                    Coef = Coef + eta * (-1*(Coef / epoch))
        return Coef

    #for convergence
    def updateCoefTillConvergence(self, X_train, y_train,epsilon,learning_factor):
        Coef_prev = self.Coef[:,np.newaxis]
        Coef_new = Coef_prev
        num_samples = X_train.shape[0]

        # Running a loop until convergence
        while True:

            hypothesis_prev = sigmoid_vectorized(np.matmul(X_train, Coef_prev))
            #print("STARTS",Coef_prev.shape)
            #print(Coef_new.shape)
            #print(hypothesis_prev.shape)
            loss_prev = -np.sum(np.multiply(y_train, np.log(hypothesis_prev)) + np.multiply((1-y_train), np.log(1-hypothesis_prev)))/num_samples
            #print(loss_prev.shape)
            #print(X_train.shape)
            #print(y_train.shape)
            #print((y_train-hypothesis_prev))
            gradient = -np.matmul(X_train.T, (y_train-hypothesis_prev))
            #print(gradient.shape)
            delta = - (learning_factor*gradient)/num_samples
            #print(delta.shape)
            # updating Coefficient using delta
            Coef_new = Coef_prev + delta
            hypothesis_new = sigmoid_vectorized(np.matmul(X_train, Coef_new))
            #print(hypothesis_new.shape)
            loss_new = -np.sum(np.multiply(y_train, np.log(hypothesis_new)) + np.multiply((1-y_train), np.log(1-hypothesis_new)))/num_samples
            #print(loss_new.shape)
            if (abs(loss_new-loss_prev))< epsilon:
                break
            Coef_prev = Coef_new

        # return Coefficient
        return Coef_new

    def predict(self, X_test):
        total = len(X_test)
        yPredicted = np.zeros(total)
        x0 = np.ones((total,))
        i = 0
        X_test_normalized =  (X_test - self.mean) / self.std
        X_test_new = np.insert(X_test_normalized, 0, x0, axis=1)
        for row in X_test_new:
            yhat = self.predicting(row, self.coef)
            #print(yhat)
            yhat = sign(yhat)
            yPredicted[i] = yhat
            i = i + 1
        return yPredicted

    # Make a prediction with coefficients
    def predicting(self, row, coefficients):
        yhat = coefficients[0]
        #print(coefficients)
        #print(row)
        #print(len(row))
        #print(yhat)
        for i in range(len(row) - 1):
            yhat += coefficients[i] * row[i]
            #print(yhat)
        ans = 1.0 / (1.0 + math.exp(-yhat))

        return ans

