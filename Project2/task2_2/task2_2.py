# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plotData2D(X, filename=None):
    # creating a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # setting x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)

    # setting the properties of the legend of the plot
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

    # reading data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # reading height and weight data into 2D array
    X = data[:,0:2].astype(np.float)
    
    
    # transposing the data matrix 
    X = X.T

    # for plotting height vs. weight 
    # copying information rows of X into 1D arrays of height and weight
    h = np.copy(X[1,:])
    w = np.copy(X[0,:])
    
    # creating a new data matrix Z by stacking h and w
    Z = np.vstack((h,w))

    # plotting this representation of the data
    plotData2D(Z, 'plotHW.pdf')
    
     # read height data into 2D array (i.e. into a matrix)  --remove
    #X = data[:, 1].astype(np.float)     --remove

    # Sample mean and standard deviation for heights to parameterize a normal distribution
    h_mean = np.mean(h)
    h_std = np.std(h)
    
    #defining fnction for removing outliers in weight data for finding correct mean and standard deviation of available data
    def reject_outliers(data):
     filtered = [e for e in data if e>0]
     return filtered
    
    #array with outliers removed
    w=reject_outliers(w)
       
    # Sample mean and std calculation for weight after removing outliers
    w_mean = np.mean(w)
    w_std = np.std(w)
    
                
    #model for height vs weight data using simple linear regression
    wd = np.copy(X[0,:])
    wd = [w_mean if i < 0 else i for i in wd]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split (h,wd,test_size=0.2,random_state=0)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    from sklearn.linear_model import LinearRegression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train,y_train)
    plt.scatter(h,wd,color='red')
    plt.scatter(X_test,linear_reg.predict(X_test))
    plt.title('Lin Reg model for height vs weight data')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    
    
    
    #finding predicted values of heights having negative weights
    """
    XT = X.T
    print(XT)
    def get_outliers(data):
     filtered = [data[:, np.argwhere(np.all(data[..., :] < 0, axis=0))] for e in data[:,0] if e>0]
     return filtered
       
    w_outliers=get_outliers(Z)
    print(Z[:, np.argwhere(np.all(Z[..., :] > 0, axis=0))])  
    V = Z[:, np.argwhere(np.all(Z[..., :] < 0, axis=0))] """
    
    # Plotting the height data as circle markers
    plt.plot(X, np.zeros_like(X) + 0, 'o', color='b', alpha=0.5, label='data')

    # Plotting the Gaussian fit for height data
    xmin, xmax = 140, 210
    x = np.linspace(xmin, xmax, 1000) # 1000 evenly spaced numbers over xmin and xmax for plotting normal distribution

    
    
    from scipy.stats import norm
    p = norm.pdf(x, h_mean, h_std) # specifying a normal distribution for sample h_mean and h_std
    plt.plot(x, p, 'k', linewidth=2, color='y', label='normal') # Plot the gaussian

    plt.ylim(0,0.06)
    #specifying title and plotting title and legend
    title = "Univariate Gaussian Distribution fit to: mu = %.2f,  std = %.2f" % (h_mean, h_std)
    plt.title(title)
    plt.legend()

    #Saving the figure
    plt.savefig("gaussian_fit_to_height.pdf",  # facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)

    # Display figure
    plt.show()
       
    
    
    from numpy import cov
    from scipy.stats import multivariate_normal as mvn
    Sigma = cov(h,wd)[0,1]
    cov = np.array([[h_std,Sigma],[Sigma,w_std]])
    mu = np.array([h_mean, w_mean])
    r = mvn.rvs(mean=mu, cov=cov, size=1000)
    
    plt.scatter(r[:,0],r[:,1])
    #plt.axis('equal')
    title = "Bivariate Gaussian Distribution" 
    plt.title(title)
    plt.xlabel('height')
    plt.ylabel('weight')
        
    
    #Saving the fig
    plt.savefig("gaussian_weight_fit_to_height.pdf",  # facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)

    # Display figure
    plt.show()

    
    
    
    
