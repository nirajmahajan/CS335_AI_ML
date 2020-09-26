import numpy as np
import argparse

def get_data(dataset):
	datasets = ['D1', 'D2']
	assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
	X_train = np.loadtxt(f'data/{dataset}/training_data')
	Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
	X_test = np.loadtxt(f'data/{dataset}/test_data')
	Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

	return X_train, Y_train, X_test, Y_test

def get_features(x):
	'''
	Input:
	x - numpy array of shape (2500, )

	Output:
	features - numpy array of shape (D, ) with D <= 5
	'''
	### TODO
	xin = x.copy()
	assert(x.max() <= 1)

	# thresholding
	xin[xin < 0.4] = 0
	xin[xin >= 0.4] = 1

	
	image = xin.reshape(50,50)
	indices = image == 1
	indices_r = np.indices((50,50))[0][indices]
	indices_c = np.indices((50,50))[1][indices]
	max_rad = ((indices_c-25)**2 + (indices_r-25)**2).max()**0.5

	# generate a circular mask
	Ymask, Xmask = np.ogrid[:50, :50]
	dist_from_center = np.sqrt((Xmask-25)**2 + (Ymask-25)**2)
	cm1 = dist_from_center >= max(max_rad - 6,4)
	cm2 = dist_from_center <= max_rad
	circular_mask = np.logical_and(cm1,cm2)


	# feature1 perimeter
	mask = np.array([1,1,1,1,1])
	perimeter_mask = np.convolve(xin,mask)
	perimeter = 0
	perimeter += (perimeter_mask == 2).sum()
	perimeter += (perimeter_mask == 3).sum()
	perimeter_mask = np.convolve(xin.reshape(50,50).T.reshape(-1),mask)
	perimeter += (perimeter_mask == 2).sum()
	perimeter += (perimeter_mask == 3).sum()

	# feature2 = area
	area = xin.sum()

	# feature3 = ring area
	circle_white = (xin.reshape(50,50)[circular_mask] == 0).sum()



	ans = np.array([perimeter, area, circle_white])
	return ans

	### END TODO

class Perceptron():
    def __init__(self, C, D):
        '''
        C - number of classes
        D - number of features
        '''
        self.C = C
        self.weights = np.zeros((C, D))
        
    def pred(self, x):
        '''
        x - numpy array of shape (D,)
        '''
        ### TODO: Return predicted class for x
        return np.argmax(self.weights.dot(x.reshape(-1,1)))
        ### END TODO

    def train(self, X, Y, max_iter=200):
        for _ in range(max_iter):
            for i in range(X.shape[0]):
                ### TODO: Update weights
                yhat = self.pred(X[i,:])
                if yhat == Y[i]:
                    continue
                else:
                    self.weights[yhat,:] -= X[i,:]
                    self.weights[Y[i],:] += X[i,:]
                ### END TODO

    def eval(self, X, Y):
        n_samples = X.shape[0]
        correct = 0
        for i in range(X.shape[0]):
            if self.pred(X[i]) == Y[i]:
                correct += 1
        return correct/n_samples

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data('D2')

	X_train = np.array([get_features(x) for x in X_train])
	X_test = np.array([get_features(x) for x in X_test])

	C = max(np.max(Y_train), np.max(Y_test))+1
	D = X_train.shape[1]

	perceptron = Perceptron(C, D)

	perceptron.train(X_train, Y_train)
	acc = perceptron.eval(X_test, Y_test)
	print(f'Test Accuracy: {acc}')
