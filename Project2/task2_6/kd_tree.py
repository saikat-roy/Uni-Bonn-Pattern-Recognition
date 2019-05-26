import numpy as np

class KDTree:

    def __init__(self, k):
        self.k = k
        self.last_dim = None
        self.X = None
        self.Y = None
        self.root = None

    def train(self, x, y, dim_mode='alternate', split_mode='mid'):
        """
        :param x: ndarray
        :param y: ndarray
        :param dim_mode:
        :param split_mode:
        :return:
        """
        self.X = x
        self.Y = y
        idxs = np.array([i for i in range(self.X.shape[0])])
        self.root = self._train(idxs=idxs, dim_mode=dim_mode, split_mode=split_mode)

    def _train(self, idxs, dim_mode='alternate', split_mode='mid'):

        dim = self.select_dim(dim_mode=dim_mode, idxs=idxs if dim_mode == 'var' else None)
        split_point = self.select_split(split_mode=split_mode, split_dim=dim, idxs=idxs)

        if idxs.shape[0]==1:
            return KDNode(k=None, val=None, idx=idxs)

        # Only if there is a split to be made
        node = KDNode(k=dim, val=split_point)

        left_idxs = idxs[self.X[idxs]<=split_point]
        node.left_node = self._train(idxs=left_idxs, dim_mode=dim_mode, split_mode=split_mode) if left_idxs.shape[0]>0 \
                                else None

        right_idxs = idxs[self.X[idxs] > split_point]
        node.right_node = self._train(idxs=right_idxs, dim_mode=dim_mode, split_mode=split_mode) if \
                            right_idxs.shape[0] > 0 else None

        return node

    def select_dim(self, dim_mode, idxs=None):
        if dim_mode == 'alternate':
            if self.last_dim is None:
                self.last_dim = 0
                return self.last_dim
            else:
                self.last_dim = (self.last_dim+1)%self.k
        elif dim_mode == 'var': # For dim of highest variance
            return np.argmax(np.var(self.X[idxs], axis=0))
        else:
            raise ValueError # Choose 'alternate' or 'var' for dimension choice

    def select_split(self, split_mode, split_dim, idxs):
        if split_mode == 'mid':
            return np.mean(self.X[idxs, split_dim], axis=0)
        elif split_mode == 'median':
            return np.median(self.X[idxs, split_dim], axis=0)
        else:
            raise ValueError # Choose split point as 'mid' or 'median' point of data

    def __str__(self):
        raise NotImplementedError
        print(self.root.val, self.root.k)

    def plot(self):
        assert self.k == 2
        raise NotImplementedError
    

class KDNode:

    def __init__(self, k, val, idx):

        self.k = k
        self.val = val
        self.idx = idx
        self.left_node = None
        self.right_node = None