import numpy as np
import matplotlib.pyplot as plt
from utils import load_data1, split_data

def mse(X, Y, w, b):
    '''
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, 1)
    Y - numpy array of shape (n_samples, 1)
    w - a float
    b - a float
    '''

    ## TODO
    n_samples = X.shape[0]
    mse = (1./(2*n_samples)) * ((X*w + b - Y)**2).sum()
    ## END TODO

    return mse

def ordinary_least_squares(X_train, Y_train, X_test, Y_test, lr=0.001, max_iter=200):
    train_mses = []
    test_mses = []

    # Initialize w and b
    ## TODO
    train_len = X_train.shape[0]
    n_features = X_train.shape[1]
    w = np.random.normal(1)
    b = 0
    ## END TODO

    for i in range(max_iter):
        ## TODO: Compute train and test MSE
        train_mse = mse(X_train,Y_train,w,b)
        test_mse = mse(X_test,Y_test,w,b)
        ## END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        ## TODO: Update w and b using a single step of gradient descent
        gradient_w = (X_train*(w*X_train + b - Y_train)).sum()/(train_len)
        gradient_b = (w*X_train + b - Y_train).sum()/(train_len)
        
        w = w - lr*gradient_w
        b = b - lr*gradient_b
        ## END TODO

    return w, b, train_mses, test_mses

if __name__ == '__main__':
    # Load and split data
    X, Y = load_data1('data1.csv')
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    w, b, train_mses, test_mses = ordinary_least_squares(X_train, Y_train, X_test, Y_test)

    # Plots
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.plot(train_mses)
    plt.plot(test_mses)
    plt.legend(['Train MSE', 'Test MSE'])
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.subplot(122)
    plt.plot([-20, 50], [-20*w+b, 50*w+b], color='r')
    plt.scatter(X_train, Y_train, color='b', marker='.')
    plt.scatter(X_test, Y_test, color='g', marker='x')
    for x, y in zip(X_test, Y_test):
        plt.plot([x, x], [w*x+b, y], color='gray', zorder=-1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()