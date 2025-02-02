import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_data(dataset,num_train_samples=-1):
    datasets = ['D1', 'D2']
    assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
    X_train = np.loadtxt(f'data/{dataset}/training_data')
    Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
    X_test = np.loadtxt(f'data/{dataset}/test_data')
    Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

    return X_train[:num_train_samples,:], Y_train[:num_train_samples], X_test, Y_test

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

    def train(self, X, Y, max_iter=2):
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
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--num_samples', type=int, default=-1,
                    help='Number of samples to train on')
    args = parser.parse_args()
    
    num_train_samples = args.num_samples

    X_train, Y_train, X_test, Y_test = get_data('D1',num_train_samples)

    C = max(np.max(Y_train), np.max(Y_test))+1
    D = X_train.shape[1]

    perceptron = Perceptron(C, D)

    perceptron.train(X_train, Y_train)
    acc = perceptron.eval(X_test, Y_test)
    print(f'Test Accuracy: {acc}')

    # plot Q3a
    print('Solving part 3a')
    test_errors = []
    n_samples_list = [10, 100, 1000, 10000, 50000, 80000]

    for i,ns in enumerate(n_samples_list):
        print('Computing for n_samples = {}'.format(ns))
        num_train_samples = ns

        X_train, Y_train, X_test, Y_test = get_data('D1',num_train_samples)

        C = max(np.max(Y_train), np.max(Y_test))+1
        D = X_train.shape[1]

        perceptron = Perceptron(C, D)

        perceptron.train(X_train, Y_train)
        acc = perceptron.eval(X_test, Y_test)
        test_errors.append(1-acc)

    # Plots
    plt.figure(figsize=(4,4))
    plt.plot(n_samples_list, test_errors, color = 'blue', marker = '.')
    plt.xlabel('Number of samples (log scale)')
    plt.xscale('log')
    plt.ylabel('Test errors')
    plt.savefig('./figs/q3a.png')
    plt.show()