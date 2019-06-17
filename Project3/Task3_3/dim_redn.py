import numpy as np

if __name__ == "__main__":

    X = np.loadtxt('data-dimred-X.csv', dtype=np.float, delimiter=',')
    y = np.loadtxt('data-dimred-y.csv', dtype=np.float)

    print(X.shape, y.shape)