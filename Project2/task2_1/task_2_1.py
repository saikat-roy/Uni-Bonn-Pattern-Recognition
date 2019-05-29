import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')

    # plot the data
    axs.plot(X[0, :], X[1, :], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0, :].min()
    xmax = X[0, :].max()
    axs.set_xlim(xmin - 10, xmax + 10)
    axs.set_ylim(-2, X[1, :].max() + 10)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data])

    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:, 0:2].astype(np.float)
    # read gender data into 1D array (i.e. into a vector)
    y = data[:, 2]

    # let's transpose the data matrix
    X = X.T

    # now, plot weight vs. height using the function defined above
    plotData2D(X, 'plotWH.pdf')

    # next, let's plot height vs. weight
    # first, copy information rows of X into 1D arrays
    w = np.copy(X[0, :])
    h = np.copy(X[1, :])

    index = np.where(w<0)
    #print(index)
    h = np.delete(h, index)
    #print(len(h))

    w = w[w>=0]
    #print(len(w))
    row=[]
    def get_matrices(w,h,n):
        X1 = np.zeros((len(h),n+1))
        #print(X)
        for i in range(len(h)):
            for j in range(n+1):
                x=h[i]
                X1[i][j] = pow(x,j)
        print(X1)
        return X1
    X1 = get_matrices(w,h,10)
    X1 = np.mat(X1)
    inter1 = X1.transpose() * X1
    #print(inter1)
    inter2 = np.linalg.pinv(inter1)
    #id = np.identity(11)
    #inter2 = np.linalg.solve(inter1,id)
    inter3 = np.mat(inter2) * np.mat(X1.transpose())
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
    plt.xlabel('')
    plt.ylabel('Height')

    plt.show()
