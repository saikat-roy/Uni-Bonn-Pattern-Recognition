import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt


class BayesianPoly():

    def __init__(self, reg_order, sigma_0_sq):

        self.reg_order = reg_order
        self.sigma_0_sq = sigma_0_sq
        self.sigma = None
        self.chi = None
        self.mu = None
        self.X_mu = None
        self.X_sigma = None
        self.y_mu = None
        self.y_sigma = None
        self.W_map = None
        self.W_mle = None

    def fit(self, X, y):

        print(X, y)
        X = self.preprocess_X(X)
        y = self.preprocess_y(y)

        if self.y_sigma is None:
            self.y_sigma = np.std(y)

        # Fitting the weights of the model using MAP
        self.W_map = np.dot(np.dot(inv(np.dot(X, X.T) + (np.identity(X.shape[0])*(self.y_sigma/self.sigma_0_sq)**2)),X),y)

        # STILL NOT WORKING PROPERLY FOR SOME REASON
        self.W_mle = np.dot(np.dot(inv(np.dot(X, X.T)), X), y)

        # Used only for predictive distribution estimation (NOT IMPLEMENTED CURRENTLY)
        self.chi = (np.dot(X, X.T)/(self.y_sigma**2)) + (np.identity(X.shape[0])/self.sigma_0_sq)
        self.mu = (np.dot(np.dot(inv(self.chi),X),y)/(self.y_sigma**2))

    def evaluate(self, X, w_type='map'):

        X = self.preprocess_X(X)

        if w_type == 'map':
            y = np.dot(self.W_map.T, X).reshape(X.shape[1])
        elif w_type == 'mle':
            y = np.dot(self.W_mle.T, X).reshape(X.shape[1])
        return y

    def evaluate_dist(self, X):
        """
        Does not work properly yet
        :param X:
        :return:
        """
        X = self.preprocess_X(X)
        print(X.shape)

        y = np.linspace(start=50, stop=120)
        print(y)
        y_pred = []
        fig, ax = plt.subplots(figsize=(10, 10))

        #y = np.array([])
        for x in X.T:
            x= np.expand_dims(x,1)
            mean = np.dot(self.mu.T, x)
            print(mean.shape)

            print(x.T.shape, inv(self.chi).shape, x.shape)
            std = (self.y_sigma**2) + np.dot(x.T,np.dot(inv(self.chi),x))

            print(self.y_sigma**2, std.shape)

            p_y_XD = norm.pdf(y, mean, std) # specify a normal distribution for sample mean and std
            print(p_y_XD.shape)
            for i in range(p_y_XD.shape[1]):
                print(x.shape, p_y_XD[0,i])
                ax.scatter(x[1,0], p_y_XD[0,i]*120)
            y_pred.append(p_y_XD)
        print(y_pred)
        plt.show()
        return p_y_XD

    def preprocess_X(self, X):

        def f(x):
            return np.array([x[0] ** i for i in range(self.reg_order + 1)])

        X = np.expand_dims(X, axis=1)
        X = np.apply_along_axis(f, axis=1, arr=X)
        X = X.T
        return X

    def preprocess_y(self, y):
        y = np.expand_dims(y, axis=1)
        return y


def mse(y_pred, y_true):
    return np.mean((y_pred-y_true)**2)
    # return y_pred-y_true


if __name__ == "__main__":

    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)
    # Removing rows with missing weights
    data = data[data[:,0]!='-1',:]

    # read height data into 1D array (i.e. into a matrix)
    X = data[:, 1].astype(np.float)

    # read weight data into 1D array (i.e. into a vector)
    y = data[:, 0].astype(np.float)

    plt.scatter(X, y, color='black', label='Data')

    print(X.shape, y.shape)

    model = BayesianPoly(reg_order=5, sigma_0_sq=3)
    model.fit(X, y)

    X_plot = np.linspace(150, 200, num=200)

    Y_plot = model.evaluate(X_plot, w_type='map')
    plt.plot(X_plot, Y_plot, 'k', linewidth=2, color='red', label='MAP') # Plot the gaussian

    # Y_plot = model.evaluate(X_plot, w_type='mle')
    # plt.plot(X_plot, Y_plot, 'k', linewidth=2, color='green', label='MLE')  # Plot the gaussian

    title = "Bayesian Regression Fit to: Height vs Weight"
    plt.title(title)
    #plt.ylim(50,120)
    plt.legend()
    plt.savefig("bayesian_regr.pdf", facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print("MSE for MAP estimation = {}".format(mse(model.evaluate(X, w_type='map'),y)))
    # print("MSE for MLE estimation = {}".format(mse(model.evaluate(X, w_type='mle'), y)))