import numpy as np
from numpy import inf
from numpy import sqrt

from scipy.integrate import nquad   # for numerical integration
from scipy.integrate import quad    # for numerical integration


def SEnorm(k, N, n, eta, paramX, sigY=None,maxIter=None):

    """
    State evolution for AMP with Gaussian signal

    the sparse signal model: x_0 ~ (k/N)*Norm + (1-k/N)*delta_0

    Input:
        k:          number of non-zero values
        N:          signal dim
        n:          measurement dim
        eta:        the soft threshold function for AMP
        paramX = [muX, sigX] mean & std of the non-zero values
        sigY:       the std of the measurement noise
        maxIter:    the maximum number of iterations

    Output:
        sigAll:     trace of sigmas
    """

    # Default input parameters
    if sigY is None:
        sigY=0

    if maxIter is None:
        maxIter=100


    delta = n / N       # undersampling ratio
    epsilon = k / N     # sparsity

    # parameters of the X distribution
    muX = paramX[0]     # mean
    sigX = paramX[1]    # std

    # lambda function for the standard norm PDF
    normpdf = lambda  x: np.exp(-0.5* x**2)/np.sqrt(2*np.pi)

    # initial estimate
    E_xi2=(muX**2 + sigX**2) * epsilon
    sigma=sqrt(E_xi2 / delta + sigY**2)
    print('sigma='+str(sigma))
    sigAll = [sigma]            # save sigma

    # main loop starts here
    for itr in range(maxIter):
        print('itr = '+str(itr))

        ## Step 1. calc the expectation for non-zero values ~ N(muX,sigX^2)
        # 1) original form
        # funcGaussian=lambda x,z: ((eta(x + sigma*z, sigma) - x) ** 2)*normpdfGen(x,muX,sigX)*normpdf(z)
        # 2) standard normal form (x = u + sigma y, where y is standard norm)
        funcGaussian=lambda y,z: ((eta(muX + sigX*y + sigma*z, sigma) - muX - sigX*y) ** 2) * normpdf(y) * normpdf(z)
        E1=nquad(funcGaussian, [[-inf,inf],[-inf,inf]])

        ## Step 2. calc the expectation for zero values
        func0=lambda z: ((eta(sigma*z,sigma))**2)*normpdf(z)
        E2=quad(func0,-inf,inf)

        #print(E1)
        #print(E2)
        #print(epsilon)

        ## Step 3. Combine E1 and E2 according to the sparsity epsilon=k/N
        Expectation=epsilon*E1[0] + (1-epsilon)*E2[0]

        sigma = sqrt(Expectation / delta + sigY ** 2)
        print('    sigma='+str(sigma))
        sigAll.append(sigma)        # save sigma

    return sigAll

if __name__ == '__main__':
    pass
