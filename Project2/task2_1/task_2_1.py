import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



if __name__ == "__main__":
    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file

    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)
    # Removing rows with missing weights
    data = data[data[:, 0] != '-1', :]

    # read height data into 1D array (i.e. into a matrix)
    X = data[:, 1].astype(np.float)

    # read weight data into 1D array (i.e. into a vector)
    y = data[:, 0].astype(np.float)

    plt.scatter(X, y, color='black', label='Data')
    w=x
    def get_matrices(w,h,n):
        X1 = np.zeros((len(h),n+1))
        #print(X)
        for i in range(len(h)):
            for j in range(n+1):
                x=h[i]
                X1[i][j] = pow(x,j)
        print(X1)
        return X1
    X1 = get_matrices(w,h,1)
    X1 = np.mat(X1)
    X1_mu = np.mean(X1,axis=1)
    X1_sigma = np.std(X1,axis=1)
    ztrans_X1 = ((X1 - X1_mu)/ (X1_sigma + 1e-8))
    ztrans_X1[0,:] = 1.0
    inter1 = ztrans_X1.transpose() * ztrans_X1
    #print(inter1)
    inter2 = np.linalg.inv(inter1)
    #id = np.identity(11)
    #inter2 = np.linalg.solve(inter1,id)
    inter3 = np.mat(inter2) * np.mat(ztrans_X1.transpose())
    W = np.mat(inter3) * np.mat(w).transpose()
    #print(W)



    def fn(h, W):
        px = 0
        for index in range(0, np.size(W)):
            px += (W[index] * (h ** index))  # evaluate the P(x)
        return px

    px = fn(h,W)
    plt.scatter(h, w, color='blue')
    plt.plot(h, np.squeeze(np.asarray(px)), color='red')
    plt.title('Weight')
    plt.xlabel('Height')
    plt.ylabel('Weight')

    plt.show()
