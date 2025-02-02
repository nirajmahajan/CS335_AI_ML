import numpy as np


def load_data(file):
    '''
    Given a file, this function returns X, the features
    and Y, the output

    Args:
    filename - is a csv file with the format

    feature1,feature2, ... featureN,y
    0.12,0.31,Yes, ... ,5.32

    Returns:
    X - numpy array of shape (number of samples, number of features)
    Y - numpy array of shape (number of samples, 1)
    '''
    # f = open(file, )
    data = np.loadtxt(file, delimiter=',', skiprows=1, dtype='str', encoding="latin1")
    X = data[:, :-1]
    Y = data[:, -1:].astype(float)
    # TODO modify X to discard columns not relevant in predicting the success of the song.
    X = np.concatenate([data[:, :1],data[:, 5:-1]],1)
    # END TODO
    return X, Y


def split_data(X, Y):
    '''
    Split data into train and test sets
    The first floor(train_ratio*n_sample) samples form the train set
    and the remaining the test set

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    train_ratio - fraction of samples to be used as training data

    Returns:
    X_train, Y_train, X_test, Y_test
    '''

    ## TODO split the songs. Test data should contain 2010 songs and train data should contain all other songs.
    n_features = X.shape[1]
    train_indices = X[:,1] < 2010
    test_indices = X[:,1] == 2010
    X_train = X[train_indices,:]
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]
    X_test = X[test_indices,:]
    X_train = np.concatenate([X_train[:,0].reshape(-1,1),X_train[:,2:]],1)
    X_test = np.concatenate([X_test[:,0].reshape(-1,1),X_test[:,2:]],1)
    ## END TODO

    return X_train, Y_train, X_test, Y_test


def one_hot_encode(X, labels):
    '''
    Args:
    X - numpy array of shape (n_samples, 1)
    labels - list of all possible labels for current category

    Returns:
    X in one hot encoded format (numpy array of shape (n_samples, n_labels))
    '''
    X.shape = (X.shape[0], 1)
    newX = np.zeros((X.shape[0], len(labels)))
    label_encoding = {}
    for i, l in enumerate(labels):
        label_encoding[l] = i
    for i in range(X.shape[0]):
        newX[i, label_encoding[X[i, 0]]] = 1
    return newX


def normalize(X):
    '''
    Returns normalized X

    Args:
    X of shape (n_samples, 1)

    Returns:
    (X - mean(X))/std(X)
    '''
    return (X - np.mean(X)) / np.std(X)


def preprocess(X, Y):
    '''
    X - feature matrix; numpy array of shape (n_samples, n_features)
    Y - outputs; numpy array of shape (n_samples, 1)

    Convert data X obtained from load_data2 to a usable format by gradient descent function
    Use one_hot_encode() to convert

    NOTE 1: X has first column denote index of data point. Ignore that column
            and add constant 1 instead (for bias)
    NOTE 2: For categorical string data, encode using one_hot_encode() and
            normalize the other features and Y using normalize()
    '''

    Xn = np.ones((X.shape[0], 1))
    for i in range(0, X.shape[1]):
        try:
            f = X[:, i].astype(float)
            if i == 0:
                Xn = np.concatenate([Xn, f.reshape(-1, 1)], axis=1)
            else:
                Xn = np.concatenate([Xn, normalize(f.reshape(-1, 1))], axis=1)
        except ValueError:
            enc = one_hot_encode(X[:, i], np.unique(X[:, i]))
            Xn = np.concatenate([Xn, enc], axis=1)
    return Xn, Y

