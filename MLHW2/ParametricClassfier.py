import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')
class MultiGaussClassify:

    #for intializeing with values given in assignment question
    def __init__(self, k):
        self.priors = np.full(k, 1 / k)
        self.means = np.zeros((k, k))
        self.covariances = np.identity(k)
        self.num_classes = k

    #fit to model given X,y
    def fit(self, X, y, diag):
        self.priors = self.Priors(y)
        self.means = self.Means(X, y)
        self.covariances = self.Covariance(X, y, diag)
        return self

    #Compute all priors
    def Priors(self, y):
        classes, Freq = np.unique(y, return_counts=True)
        priors = Freq / y.shape[0]
        return priors

    #Compute mean for all classes
    def Means(self, X, y):
        mean_estimates = np.empty([self.num_classes, X.shape[1]])
        for i in range(self.num_classes):
            indices = np.where(y == i)[0]
            Xs = X[indices, :]
            mu_i = np.sum(Xs)
            mean_estimates[i] = mu_i/indices.shape[0]
        return mean_estimates

    #Predictlabels for X and return predicted y
    def predict(self, X):
        covDet = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            covDet[i] = np.linalg.det(self.covariances[i])
        covInv = np.empty([self.covariances.shape[0], self.covariances.shape[1], self.covariances.shape[2]])
        for i in range(self.num_classes):
            covInv[i] = np.linalg.pinv(self.covariances[i])
        yhat = np.zeros(X.shape[0])
        for object in range(X.shape[0]):
            max = -math.inf
            predictedClass = 0
            for i in range(self.num_classes):
                g = self.g(X[object], covDet, covInv, i)
                if g > max:
                    predictedclass = i
                    max = g
            yhat[object] = predictedclass
        return yhat

    #Find FUll and diagonal covariance
    def Covariance(self, X, y, diag):
        covariance = np.empty([self.num_classes, X.shape[1], X.shape[1]])
        for i in range(self.num_classes):
            indices = np.where(y == i)[0]
            num_i = indices.shape[0]
            Xs = X[indices, :]
            Xs_mu_i = (Xs - self.means[i]).transpose()
            Xs_mu_i_tr = Xs_mu_i.transpose()
            covariance_i =0.0001*np.identity(X.shape[1])+(np.matmul(Xs_mu_i, Xs_mu_i_tr)/num_i)
            size = covariance_i.shape
            mat = np.diag(np.diag(covariance_i))
            if diag:
                covariance[i] = mat
            else:
                covariance[i] = covariance_i
        return covariance

    # Discriminant function gi(x) = log p(Ci) + log p(x|Ci)
    def g(self, X, covDeterminant, covInverse, i):
        g1 = math.log(self.priors[i])-0.5*math.log(covDeterminant[i])
        g2 = -0.5*(X - self.means[i]).transpose().dot(covInverse[i]).dot(X - self.means[i])
        return g1+g2
