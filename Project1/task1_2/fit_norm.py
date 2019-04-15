import numpy as np
import matplotlib as plt

if __name__ == "__main__":
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:, 0:2].astype(np.float)

    mean = np.mean(X)
    std = np.std

    # Solution to Question 1.1: Plot the data after excluding the missing weight values
    plotData2D(X[:, X[0, :] >= 0], 'plotWH_nonegatives.pdf')
