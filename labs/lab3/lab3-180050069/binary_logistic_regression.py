import numpy as np
import argparse
from utils import *


class BinaryLogisticRegression:
    def __init__(self, D):
        """
        D - number of features
        """
        self.D = D
        self.weights = np.random.rand(D, 1)

    def predict(self, X):
        """
        X - numpy array of shape (N, D)
        """
        # TODO: Return a (N, 1) numpy array of predictions.
        a = X@self.weights
        a = 1/(1+np.exp(-a))
        return np.round(a)
        # return np.zeros(a.shape)
        # END TODO

    def train(self, X, Y, lr=0.5, max_iter=10000):
        for _ in range(max_iter):
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.
            n_samples = Y.shape[0]
            a = np.matmul(X,self.weights)
            predictions =  1/(1+np.exp(-a))
            gradient = (1/n_samples)*(np.matmul(X.T,(Y-predictions)))
            self.weights += lr*gradient

            # END TODO

            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4
            if (gradient**2).sum()**0.5 < 1e-4:
                break

            # End TODO

    def accuracy(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        accuracy = ((preds == Y).sum()) / len(preds)
        return accuracy

    def f1_score(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        # TODO: calculate F1 score for predictions preds and true labels Y
        pred_ones = preds == 1
        pred_zeros = preds == 0
        TP = (Y[pred_ones] == preds[pred_ones]).sum()
        FP = pred_ones.sum() - TP
        TN = (Y[pred_zeros] == preds[pred_zeros]).sum()
        FN = pred_zeros.sum() - TN

        recall = TP/(TP+FN+1e-16)
        precision = TP/(TP+FP+1e-16)
        return (2*recall*precision)/(recall+precision + 1e-16)


        # End TODO


if __name__ == '__main__':
    np.random.seed(335)

    X, Y = load_data('data/songs.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    D = X_train.shape[1]

    lr = BinaryLogisticRegression(D)
    # print('Debug: Train')
    lr.train(X_train, Y_train)
    preds = lr.predict(X_test)
    acc = lr.accuracy(preds, Y_test)
    f1 = lr.f1_score(preds, Y_test)
    print(f'Test Accuracy: {acc}')
    print(f'Test F1 Score: {f1}')
