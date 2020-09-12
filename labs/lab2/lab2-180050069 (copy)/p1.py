import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)


def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """

    # TODO
    n_samples = X.shape[0]
    mse = (1./(2*n_samples))*((X.dot(W)-Y)**2).sum()
    # END TODO

    return mse

# ridge regression from previous lab
def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.001, max_iter=1000):
    '''
    reg - regularization parameter (lambda in Q2.1 c)
    '''
    train_mses = []
    test_mses = []

    ## TODO: Initialize W using using random normal 
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    W = np.random.normal(size=(n_features,1))
    ## END TODO

    for i in range(max_iter):

        ## TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        ## END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        ## TODO: Update w and b using a single step of gradient descent
        gradient = (1./n_samples)*((X_train.dot(W) - Y_train).T.dot(X_train)).T + 2*reg*W
        W -= lr*gradient
        ## END TODO

    return W, train_mses, test_mses

def ista(X_train, Y_train, X_test, Y_test, _lambda=0.2, lr=0.01, max_iter=2000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    W = np.random.normal(size = (n_features,1))
    # END TODO

    for i in range(max_iter):
        # TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.
        gradient = (1./n_samples)*((X_train.dot(W) - Y_train).T.dot(X_train)).T
        W_ls = W - lr*gradient
        W_lasso = np.zeros(W_ls.shape)
        thresh = _lambda * lr
        ind = W_ls[:,0] > thresh
        if ind.sum() > 0:
            W_lasso[ind,0] = W_ls[ind,0] - thresh
        ind = W_ls[:,0] < -thresh
        if ind.sum() > 0:
            W_lasso[ind,0] = W_ls[ind,0] + thresh
        # END TODO

        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if (((W-W_lasso)**2).sum()**0.5) <= 1e-4:
            # W = W_lasso
            break
        else:
            W = W_lasso
        # End TODO

    return W, train_mses, test_mses


if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    # W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test)

    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)
    
    lambda_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6]

    ########### commented out
    # for i,lada in enumerate(lambda_list):
    #     print("Training for lambda = {}".format(lada), flush = True)
    #     if i % 3 == 0:
    #         if not i == 0:
    #             fig.tight_layout(pad=.9)
    #             plt.savefig('./figs/q1b_{}.png'.format(i//3))
    #         fig, a = plt.subplots(1,3)
    #         fig.suptitle('1.b) Lasso Regression for different lambdas')
    #     r = i % 3
    #     W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test, _lambda = lada)
    #     a[r].plot(train_mses_ista, label = "Train")
    #     a[r].plot(test_mses_ista, label = "Test")
    #     a[r].legend(loc = "upper left")
    #     a[r].set_title("lambda = {}".format(lada))
    #     a[r].set_ylabel("MSE")
    #     a[r].set_xlabel("Iterations")
    # plt.savefig('./figs/q1b_.png')
    # plt.show()
    ########### Commented out

    final_train_mses = []
    final_test_mses = []
    for i, lada in enumerate(lambda_list):
        print("Training for lambda = {}".format(lada), flush = True)
        W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test, _lambda=lada, lr=0.001, max_iter=10000)
        final_train_mses.append(train_mses_ista[-1])
        final_test_mses.append(test_mses_ista[-1])

    plt.figure(figsize=(4,4))
    plt.plot(lambda_list, final_train_mses, marker = 'o', color = 'red')
    plt.plot(lambda_list,final_test_mses, marker = 'o', color = 'blue')
    plt.legend(['Train MSEs', 'Test MSEs'])
    plt.xlabel('Lambda')
    plt.ylabel('Final MSE')
    plt.savefig('./figs/q1b.png')
    # plt.show()

    # print(final_train_mses)
    # print(final_test_mses)


    # get scatter plots
    print('Generating Scatter Plots')
    W_ridge, train_mses_ridge, test_mses_ridge = ridge_regression(X_train, Y_train, X_test, Y_test, 10)
    W_lasso, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test)
    plt.figure()
    plt.scatter(np.arange(W_ridge.shape[0]), W_ridge[:,0], color = 'blue')
    plt.xlabel('Weight Id')
    plt.ylabel('Weight Value')
    plt.title('Ridge Regression')
    plt.savefig('./figs/q1c_1.png')
    
    plt.figure()
    plt.scatter(np.arange(W_lasso.shape[0]), W_lasso[:,0], color = 'blue')
    plt.xlabel('Weight Id')
    plt.ylabel('Weight Value')
    plt.title('Lasso Regression')
    plt.savefig('./figs/q1c_2.png')

    plt.show()
    # End TODO
