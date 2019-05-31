import numpy as np
from scipy import linalg
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')





def standardize( data):
    """ Peform feature scaling
    Parameters:
    ------------
    data : numpy-array, shape = [n_samples,]

    Returns:
    ---------
    Standardized data
    """

    return (data - np.mean(data)) / (np.max(data) - np.min(data))


def hypothesis(theta, x):
    """ Compute hypothesis, h, where
    h(x) = theta_0*(x_1**0) + theta_1*(x_1**1) + ...+ theta_n*(x_1 ** n)
    Parameters:
    ------------
    theta : numpy-array, shape = [polynomial order + 1,]
    x : numpy-array, shape = [n_samples,]

    Returns:
    ---------
    h(x) given theta values and the training data
    """
    h = theta[0]
    for i in np.arange(1, len(theta)):
        h += theta[i] * x ** i
    return h



def fit( x,y, order=5):

    d = {}
    d['x' + str(0)] = np.ones([1, len(x)],dtype=np.longdouble)[0]
    for i in np.arange(1, order + 1):
        d['x' + str(i)] = x ** (i)

    d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    X = np.column_stack(d.values())
    inter1 = np.matmul(np.transpose(X), X)
        #find inverse
    theta = np.matmul(np.matmul(linalg.pinv(inter1), np.transpose(X)), y)

    return theta


def plot_predictedPolyLine(x,y,theta):
    plt.figure()
    plt.scatter(x, y, s=30, c='b')
    line = theta[0]  # y-intercept
    label_holder = []
    label_holder.append('%.*f' % (2, theta[0]))
    for i in np.arange(1, len(theta)):
        line += theta[i] * x ** i
        label_holder.append(' + ' + '%.*f' % (2, theta[i]) + r'$x^' + str(i) + '$')

    plt.plot(x, line, label=''.join(label_holder))
    plt.title('Polynomial Fit: Order ' + str(len(theta) - 1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()

def prediction(saved_heights,theta):
    px = 0
    for index in range(0, np.size(theta)):
        px += (theta[index] * (saved_heights ** index))  # evaluate the P(x)
    return px


def prediction(saved_heights,theta):
    py=[]

    for i in saved_heights:
        r=0.0
        for j in range(len(theta)):
            r = r + theta[j] * pow(i,j)
        py.append(r)
    return py




data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)
# Removing rows with missing weights
saved_row = data[data[:, 0] == '-1', :]

data = data[data[:, 0] != '-1', :]
saved_heights = saved_row[:, 1].astype(np.float)
# read height data into 1D array (i.e. into a matrix)
X = data[:, 1].astype(np.float)

# read weight data into 1D array (i.e. into a vector)
y = data[:, 0].astype(np.float)

plt.scatter(X, y, color='black', label='Data')
#newX=standardize(X)
#newy=standardize(y)
theta = fit(X,y)
h = np.array([i+1 for i in range(155,190)])
#plot_predictedPolyLine(h,y,theta)
y_missing = prediction(saved_heights,theta)
print(y_missing)
px = prediction(h,theta)
plt.scatter(X,y , color='blue')
plt.plot(h, px, color='red')
plt.title('Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()