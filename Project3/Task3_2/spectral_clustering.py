import numpy as np
from scipy.linalg import norm, eig
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == "__main__":

    # read data as 2D array of data type 'np.float'
    X = np.loadtxt('data-clustering-2.csv', dtype=np.float, delimiter=',')
    X = X.T

    beta = 4.0
    sim = np.empty((X.shape[0], X.shape[0]), dtype=np.float)
    diag = np.zeros((X.shape[0], X.shape[0]), dtype=np.float)

    for i in range(X.shape[0]):
        sim[i,:] = np.exp(-beta*norm(X[i]-X, ord=2, axis=1)**2)
        diag[i,i] = np.sum(sim[i,:])

    L = diag - sim

    w, v = eig(L)
    print(w)
    print(w.shape, v.shape)

    sorted_idxs = np.argsort(w)
    print(sorted_idxs)
    print(w[sorted_idxs])
    w, v = w[sorted_idxs], v[:,sorted_idxs]

    fv = v[:,1]
    print(fv)
    y = (fv<=0)*1
    print(y)

    color_list = ['blue', 'red']
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(color_list), alpha=0.75)
    plt.show()

    # from sklearn.cluster import KMeans, SpectralClustering
    # kmeans = SpectralClustering(n_clusters=2).fit(X)
    #
    # color_list = ['blue', 'red']
    # plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap=ListedColormap(color_list), alpha=0.75)
    # plt.show()