import numpy as np
from kernel import *
from utils import *
import matplotlib.pyplot as plt


class KernelLogistic(object):
    def __init__(self, kernel=gaussian_kernel, iterations=100,eta=0.01,lamda=0.05,sigma=1):
        self.kernel = lambda x,y: kernel(x,y,sigma)
        self.iterations = iterations
        self.alpha = None
        self.eta = eta     # Step size for gradient descent
        self.lamda = lamda # Regularization term

    def fit(self, X, y):
        ''' find the alpha values here'''
        self.train_X = X
        self.train_y = y
        self.alpha = np.zeros((y.shape[0],1))
        kernel = self.kernel(self.train_X,self.train_X)

        # TODO
        for i in range(self.iterations):
        	preds = 1. / (1 + np.exp(-kernel.T @ self.alpha))
        	gradient = kernel @ (preds + ((self.lamda/2)*self.alpha) - self.train_y.reshape(-1,1))
        	self.alpha -= self.eta*gradient
        # END TODO


    

    def predict(self, X):
        # TODO 
        # X - n_test x d, alpha = n_train, 1

        # n_train x n_test
        kernel = self.kernel(self.train_X,X)
        preds = kernel.T @ self.alpha
        return 1. / (1 + np.exp(-preds)).reshape(-1)

        # END TODO

def k_fold_cv(X,y,k=10,sigma=1.0):
    '''Does k-fold cross validation given train set (X, y)
    Divide train set into k subsets, and train on (k-1) while testing on 1. 
    Do this process k times.
    Do Not randomize 
    
    Arguments:
        X  -- Train set
        y  -- Train set labels
    
    Keyword Arguments:
        k {number} -- k for the evaluation
        sigma {number} -- parameter for gaussian kernel
    
    Returns:
        error -- (sum of total mistakes for each fold)/(k)
    '''
    # TODO 
    err_acc = 0
    partition_size = X.shape[0]//k
    for spliti in range(k):
    	drop_start = spliti*partition_size
    	drop_end = (spliti+1)*partition_size

    	X_test = X[drop_start:drop_end,:]
    	y_test = y[drop_start:drop_end]
    	X_train = np.delete(X, np.arange(drop_start, drop_end), 0)
    	y_train = np.delete(y, np.arange(drop_start, drop_end))

    	clf = KernelLogistic(gaussian_kernel, sigma = sigma)
    	clf.fit(X_train, y_train)

    	y_predict = clf.predict(X_test) < 0.5

    	mistakes = np.sum(y_predict == y_test)

    	err_acc += mistakes/k

    return err_acc
    # END TODO

if __name__ == '__main__':
    data = np.loadtxt("./data/dataset1.txt")
    X1 = data[:900,:2]
    Y1 = data[:900,2]

    clf = KernelLogistic(gaussian_kernel)
    clf.fit(X1, Y1)

    y_predict = clf.predict(data[900:,:2]) > 0.5

    correct = np.sum(y_predict == data[900:,2])
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    if correct > 92:
        marks = 1.0
    else:
        marks = 0
    print(f"You recieve {marks} for the fit function")

    errs = []
    sigmas = [0.5, 1, 2, 3, 4, 5, 6]
    for s in sigmas:  
      errs+=[(k_fold_cv(X1,Y1,sigma=s))]
    plt.plot(sigmas,errs)
    plt.xlabel('Sigma')
    plt.ylabel('Mistakes')
    plt.title('A plot of sigma v/s mistakes')
    # plt.show()
