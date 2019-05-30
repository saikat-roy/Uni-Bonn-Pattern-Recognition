import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class KDTree:

    def __init__(self, k, dim_mode='alternate', split_mode='mid',):
        self.k = k
        self.last_dim = None
        self.X = None
        self.Y = None
        self.root = None
        self.depth = None
        self.dim_mode = dim_mode
        self.split_mode = split_mode

    def fit(self, x, y, depth=3):
        """
        :param x: ndarray
        :param y: ndarray
        :param dim_mode:
        :param split_mode:
        :return:
        """
        self.X = x
        self.Y = y
        self.depth = depth
        idxs = np.array([i for i in range(self.X.shape[0])])
        self.root = self._fit(idxs=idxs, depth=0)

    def _fit(self, idxs, depth):

        dim = self.select_dim(idxs=idxs if self.dim_mode == 'var' else None)
        split_point = self.select_split(split_dim=dim, idxs=idxs)

        if depth==self.depth:
            return KDNode(split_dim=None, val=None, idxs=idxs)

        # Only if there is a split to be made
        node = KDNode(split_dim=dim, val=split_point)

        print(idxs.shape, dim, (self.X[idxs,dim]<=split_point).shape)
        # exit()
        left_idxs = idxs[self.X[idxs,dim]<=split_point]
        node.left_node = self._fit(idxs=left_idxs, depth=depth+1) if left_idxs.shape[0]>0 else None

        right_idxs = idxs[self.X[idxs, dim] > split_point]
        node.right_node = self._fit(idxs=right_idxs, depth=depth+1) if right_idxs.shape[0] > 0 else None

        return node

    def select_dim(self, idxs=None):
        if self.dim_mode == 'alternate':
            if self.last_dim is None:
                self.last_dim = 0
            else:
                self.last_dim = (self.last_dim+1)%self.k
            return self.last_dim

        elif self.dim_mode == 'var': # For dim of highest variance
            return np.argmax(np.var(self.X[idxs], axis=0))
        else:
            raise ValueError # Choose 'alternate' or 'var' for dimension choice

    def select_split(self, split_dim, idxs):
        # To implement: Point closest to split point and not point itself.
        if self.split_mode == 'mid':
            return np.mean(self.X[idxs, split_dim], axis=0)
        elif self.split_mode == 'median':
            return np.median(self.X[idxs, split_dim], axis=0)
        else:
            raise ValueError # Choose split point as 'mid' or 'median' point of data

    def __str__(self):
        raise NotImplementedError
        print(self.root.val, self.root.k)

    def evaluate(self):
        return

    def plot(self):
        assert self.k == 2
        raise NotImplementedError


class KDNode:

    def __init__(self, split_dim, val, idxs=None):

        self.split_dim = split_dim
        self.val = val
        self.idxs = idxs # Only populated for leaf nodes
        self.left_node = None
        self.right_node = None


if __name__ == "__main__":

    with open("data2-train.dat", "r") as f:
        d = np.loadtxt(f)

    X = d[:,0:2]
    Y = d[:,2]
    Y[Y==-1] = 0
    color_list = ['blue','red']
    # color_list = ['gray','black']
    #print(Y)

    plt.scatter(X[:,0], X[:,1], c=Y, cmap=ListedColormap(color_list), alpha=0.75)
    plt.xlim((min(X[:, 0]), max(X[:, 0])))
    plt.ylim((min(X[:, 1]), max(X[:, 1])))
    # plt.show()


    model = KDTree(k=2)
    model.fit(x=X, y=Y, depth=3)

