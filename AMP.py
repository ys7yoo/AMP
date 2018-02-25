import numpy as np
from numpy import zeros
from numpy import ones

from numpy import dot
from numpy import sqrt
from numpy import abs
from numpy import sign

def norm(x):
    return np.linalg.norm(x, ord=2)

def AMP(y,Afunc,ATfunc,Eta,Etader,niter=100,saveAllEst=0):

    # Inputs:
#   y :          observations
#   Afunc :      a function handle that represents matrix A, Afunc(x,1) means
#                A*x
#   Afunc :      a function handle that represents matrix A', ATfunc(x,1) means
#                A'*x
#   Eta :        a function handle which is a generic denoiser,
#                xhat=Eta(temp,sigma)
#   Etader :     a function handle which is the derivative function of the
#                denoise function Eta, if you can't provide this derivative
#                function, please input "Null"
#   niter :      the maximum number of iterations
#                denotes how many iteration you want AMP to run, if you
#                input a positive integer t, then AMP runs t times
#                iterations for you, if you input the string 'Auto', then
#   saveAllEst:         Whether we need all the estimates in whole process (1) or
#                just the final estimate (0)

    # Related functions: Eta_der_Estimate

    # check: @A,@Eta,@Etader, sigma is the estimated std instead of variance.
# sigma_w seems useless in this function

    n=len(y)
    lengthN=ATfunc(zeros([n,1]))
    N=len(lengthN)

    ## check norms of A's columns
    # pick=randperm(N)
    # DtmIndex=pick[1:5]
    # DtmNorms=zeros(5,1)
    # I=eye(N)
    # for i in arange(1,5).reshape(-1):
    #     DtmNorms[i]=norm(Afunc[I[:,DtmIndex[i]]]) ** 2
    #
    # if (sum(DtmNorms > 1.1) + sum(DtmNorms < 0.9)) > 0:
    #     disp('It is necessary to normalize the A matrix, please wait...')
    #     tempA=zeros(n,N)
    #     tempA_ra=zeros(n,N)
    #     normalize_time_total=0
    #     for j in arange(1,N).reshape(-1):
    #         t0=copy(cputime)
    #         tempA[:,j]=Afunc[I[:,j]]
    #         normalize_time=(cputime - t0) / 60
    #         normalize_time_total=normalize_time_total + normalize_time
    #         if j / (N / 100) == fix(j / (N / 100)):
    #             normalize_time_remain=dot(normalize_time_total,(N - j)) / j
    #             percent=j / (N / 100)
    #             disp(cat('Normalizing has been through ',num2str(percent),'%.',10,'The estimated remaining time for Normalzing is ',num2str(normalize_time_remain),' minutes.'))
    #     for j in arange(1,N).reshape(-1):
    #         tempA_ra[:,j]=tempA[:,j] - mean(tempA[:,j])
    #     colnormA=(sqrt(sum(tempA_ra ** 2,1))).T
    #     ind=find(colnormA == 0)
    #     colnormA[ind]=(sqrt(sum(abs(tempA[:,ind]) ** 2,1))).T
    #     disp('Normalizing ends, Iteration starting...')
    # else:
    #     disp('It is not necessary to normalize the A matrix, Iteration starting...')
    #     colnormA=ones(N,1)
    colnormA = ones((N,1))

    # Denote normalized A matrix as AA, then, when we calculate AA*v, we need
# to do A*(v./colnormA); when we calculate AA'*v, we need to do (A'*v)./colnormA.

    #####
    # to save values for all iterations
    empiricalSigma = []
    xAll = []

    # initial estimate
    mx = zeros([N,1])
    mz = y - Afunc(mx / colnormA)

    # main loop starts here
    for iter in range(niter):
        # disp(cat('iteration = ',num2str(iter)))
        temp_z = ATfunc(mz) / colnormA + mx
        sigma_hat = norm(mz) / sqrt(n)
        mx = Eta(temp_z,sigma_hat)
        mz = y - Afunc(mx / colnormA) + mz * sum(Etader(temp_z,sigma_hat)) / n

        empiricalSigma.append(sigma_hat)
        xAll.append(mx / colnormA)

        if abs(sigma_hat - norm(mz) / sqrt(n)) < 0.001:
            break

    empiricalSigma.append(norm(mz) / sqrt(n))
    if iter == 100:
        print('Iteration reaches the maximum (100) times,\\nthe algorithm does not converge within 100 iterations.\\n')

    # if niter==100
    #empiricalSigma=empiricalSigma[1:(iter + 1)]
    #xAll=xAll[:,1:(iter + 1)]
    # end

    if saveAllEst:
        #xhat=copy(xAll)
        xhat=xAll
    else:
        #xhat=xAll[:,end()]
        xhat=xAll[-1]

    return xhat, empiricalSigma

if __name__ == '__main__':
    pass
