import numpy as np


def train_test_split(X, y, ratio=0.8):
    # Here we generate a random permutation
    indices = np.random.permutation(X.shape[0])
    
    # Here we calculate the number of objects in the train set
    train_len = int(X.shape[0] * ratio)

    # Here we split the data into train and test sets
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]

    # Here we return the train and test sets
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    wines = np.genfromtxt('wine.csv', delimiter=',')
    # Here the data is split into objects and classes
    # to process them separately. Note that an object and
    # its corresponding class will have the same index.
    X, y = wines[:, 1:], np.array(wines[:, 0], dtype=np.int32)
    # Here we call our function to look at the result.
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    # It is convenient to add visualization (a console output at least)
    # for each of the functions you add in a new step. Like this:
    print("X_train: ", "\n")
    print(X_train)
    print("y_train: ")
    print(y_train)
    print("X_test", "\n")
    print(X_test)
    print("y_test", "\n")
    print(y_test)
