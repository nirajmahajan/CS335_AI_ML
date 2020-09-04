import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)

def mse(X, Y, W):
	'''
	Compute mean squared error between predictions and true y values

	Args:
	X - numpy array of shape (n_samples, n_features)
	Y - numpy array of shape (n_samples, 1)
	W - numpy array of shape (n_features, 1)
	'''

	## TODO
	n_samples = X.shape[0]
	mse = (1./(2*n_samples))*((X.dot(W)-Y)**2).sum()
	## END TODO

	return mse

def ordinary_least_squares(X_train, Y_train, X_test, Y_test, lr=0.01, max_iter=3000):
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
		gradient = (1./n_samples)*((X_train.dot(W) - Y_train).T.dot(X_train)).T
		W -= lr*gradient
		## END TODO

	return W, train_mses, test_mses

def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.0001, max_iter=2000):
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
		reg_term = reg*(W**2).sum()
		train_mse = mse(X_train, Y_train, W) + reg_term
		test_mse = mse(X_test, Y_test, W) + reg_term
		## END TODO

		train_mses.append(train_mse)
		test_mses.append(test_mse)

		## TODO: Update w and b using a single step of gradient descent
		gradient = (1./n_samples)*((X_train.dot(W) - Y_train).T.dot(X_train)).T + 2*reg*W
		W -= lr*gradient
		## END TODO

	return W, train_mses, test_mses

def weighted_regression(X, Y, r):
	'''
	Fill up this function for problem 3.
	Use closed form expression.
	r_train is a (n,) array, where n is number of training samples
	'''

	## TODO

	## END TODO
	return W

if __name__ == '__main__':
	# Load and split data
	X, Y = load_data2('data2.csv')
	X, Y = preprocess(X, Y)
	X_train, Y_train, X_test, Y_test = split_data(X, Y)

	W, train_mses, test_mses = ordinary_least_squares(X_train, Y_train, X_test, Y_test)
	# W_ridge, train_mses, test_mses = ridge_regression(X_train, Y_train, X_test, Y_test, 10)

	# Plots
	plt.figure(figsize=(4,4))
	plt.plot(train_mses)
	plt.plot(test_mses)
	plt.legend(['Train MSE', 'Test MSE'])
	plt.xlabel('Iteration')
	plt.ylabel('MSE')
	plt.show()

