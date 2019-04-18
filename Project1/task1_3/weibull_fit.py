import math
import numpy as np

import matplotlib.pyplot as plt

class Weibull:

    def __init__(self, kappa=1, alpha=1):
        self.k_alp = np.array([kappa, alpha])

    def fit(self, D, iters = 20):
        N = D.shape[0]

        for iter in range(iters):

            term1, term2, term3, term4 = 0,0,0,0

            for i in range(h.shape[0]):
                #print(D[i], D[i]/self.k_alp[1])
                #print(math.log2(D[i]/self.k_alp[1]))
                term1 += math.log(D[i])
                term2 += (math.pow(D[i]/self.k_alp[1],self.k_alp[0]))*(math.log(D[i]/self.k_alp[1]))
                term3 += ((D[i]/self.k_alp[1])**self.k_alp[0])
                term4 += ((D[i]/self.k_alp[1])**self.k_alp[0])*((math.log(D[i]/self.k_alp[1])**2))
            print(term1,term2,term3,term4)

            dL_dk = (N / self.k_alp[0]) - (N * math.log(self.k_alp[1])) + term1 - term2
            dL_dalpha = self.k_alp[0] / self.k_alp[1] * (term3 - N)
            d2L_dk2 = -(N / (self.k_alp[0] ** 2)) - term4
            d2L_dalpha2 = (self.k_alp[0] / (self.k_alp[1] ** 2)) * (N - ((self.k_alp[0]+1)*term3))
            d2L_dkdalpha = ((1 / self.k_alp[1]) * term3) + ((self.k_alp[0]/self.k_alp[1])*term2) - (N/self.k_alp[1])

            self.k_alp = self.k_alp + \
                         np.dot(np.linalg.inv(np.array([[d2L_dk2, d2L_dkdalpha],[d2L_dkdalpha, d2L_dalpha2]])) ,
                         np.array([-dL_dk, -dL_dalpha]))

            print(self.k_alp)
        return


    def weibull(self, x):

        return (self.k_alp[0]/self.k_alp[1])*(math.pow(x/self.k_alp[1],(self.k_alp[0]-1))) * \
               np.exp(-1*math.pow(x/self.k_alp[1],self.k_alp[0]))

    def pred(self, x):
        return np.array([self.weibull(x_i) for x_i in x])

    def fit_hist(self, h, iters = 20):

        N = np.sum(h)
        for iter in range(iters):

            term1, term2, term3, term4 = 0,0,0,0

            for i in range(h.shape[0]):
                term1 += (math.log(i+1) * h[i])
                term2 += h[i]*(math.pow(i/self.k_alp[1],self.k_alp[0]))*(math.log((i+1)/self.k_alp[1]))
                term3 += h[i]*(math.pow(i/self.k_alp[1],self.k_alp[0]))
                term4 += h[i]*(math.pow(i/self.k_alp[1],self.k_alp[0]))*((math.log((i+1)/self.k_alp[1]))**2)
            print(term1,term2,term3,term4)

            dL_dk = (N / self.k_alp[0]) - (N * math.log(self.k_alp[1])) + term1 - term2
            dL_dalpha = (self.k_alp[0] / self.k_alp[1]) * (term3 - N)
            d2L_dk2 = -(N / (self.k_alp[0] ** 2)) - term4
            d2L_dalpha2 = (self.k_alp[0] / (self.k_alp[1] ** 2)) * (N - ((self.k_alp[0] + 1) * term3))
            d2L_dkdalpha = ((1 / self.k_alp[1]) * term3) + ((self.k_alp[0]/self.k_alp[1])*term2) - (N/self.k_alp[1])
            print(dL_dk,dL_dalpha, d2L_dk2,d2L_dalpha2,d2L_dkdalpha)

            self.k_alp = self.k_alp + \
                         np.dot(np.linalg.inv(np.array([[d2L_dk2, d2L_dkdalpha],[d2L_dkdalpha, d2L_dalpha2]])) ,
                         np.array([-dL_dk, -dL_dalpha]))

            print(self.k_alp)


if __name__ == "__main__":

    data = np.loadtxt('myspace.csv', dtype=np.object, comments='#', delimiter=',')
    X = data[:,1].astype(np.float)
    print(X)
    h = X[np.argmin((X==0)*1):]
    n = np.arange(1,h.shape[0]+1)
    print(h,h.shape[0],n)
    plt.plot(h)
    plt.xlim(0, X.shape[0])

    D = []
    for i in h:
        D.extend([i for j in range(int(i))])

    D = np.array(D)
    #np.random.shuffle(D)
    print(D)

    #weib = Weibull(kappa=2.81, alpha=215.43)
    weib = Weibull()
    weib.fit(D)
    # weib.fit_hist(h)

    x = np.linspace(1, X.shape[0], 1000)  # 1000 evenly spaced numbers over xmin and xmax for plotting normal distribution
    p = weib.pred(x)  # specify a normal distribution for sample mean and std
    print(sum(p))
    plt.plot(x, p*17293, 'k', linewidth=2, color='y', label='normal')  # Plot the gaussian

    #plt.ylim(0, 0.06)
    # specifying title and plotting title and legend
    #title = "Univariate Gaussian Distribution fit to: mu = %.2f,  std = %.2f" % (mean, std)
    #plt.title(title)
    plt.legend()

    # Saving the figure
    #plt.savefig("gaussian_fit_to_height.pdf",  # facecolor='w', edgecolor='w',
    #            papertype=None, format='pdf', transparent=False,
    #            bbox_inches='tight', pad_inches=0.1)

    # Display figure
    plt.show()

    #plt.show()