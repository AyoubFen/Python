#!/usr/bin/env python3

"""
A graphic comparaison of MonteCarlo variance reduction methods.

We use these methods in this script to calculate the integral of e^(x^2) between 0 and 1 (≈ 1.463) and compare the convergence and standard deviation of 4 MonteCarlo estimators:
- Crude MonteCarlo ( no variance reduction)
- Antithetic variables variance reduction
- Control variates variance reduction
- Importance sampling variance reduction


22/11/2022 ,AyoubFen
"""

import numpy as np
import matplotlib.pyplot as plt



def f(x):
    """
    -the function we want to calculate the integral of.
    -input:
        x    : scalar or numpy array if all operations used in f are vectorized
    -output:
        f(x) : scalar or numpy array depending on input.
    """
    return np.exp(x**2)


def mc_crude(n,f):
    """
    -the most basic montecarlo estimator, without any variance reduction method.
    -input:
        n : numbre of montecarlo iterations.
        f : function we want to calculate the integral of.
    -output:
        np.mean(fU)       : estimation of integral of f from 0 to 1.
        np.std(fU,ddof=1) : standard deviation of the estimation.
    """
    U = np.random.uniform(0,1,n)
    fU = f(U)
    return np.mean(fU),np.std(fU,ddof=1)


def mc_antithetic(n,f):
    """
    -montecarlo estimators with antithetic variables: we use random variables of same distribution but negatively correlated (in this case, U and 1-U).The new unbiased estimator is (f(U)+f(1-U))/2.0
    -input:
        n : numbre of montecarlo iterations.
        f : function we want to calculate the integral of.
    -output:
        np.mean(fU)       : estimation of integral of f from 0 to 1.
        np.std(fU,ddof=1) : standard deviation of the estimation.
    """
    U = np.random.uniform(0,1,n//2)
    fU = (f(U)+f(1-U))/2.0
    return np.mean(fU),np.std(fU,ddof=1)

def h(x):
    """
    -a function highly correlated to function f.
    -input:
        x    : scalar or numpy array if all operations used in h are vectorized
    -output:
        h(x) : scalar or numpy array depending on input.
    """
    return 1+x**2

def mc_control_variate(n,f,h,meanh):
    """
    -montecarlo estimators with control variate: we use a function h that is very correlated to f and that has a known integral . The new unbiased estimator is
     f(U)+ c*(h(U) - mean_h), where optimal value of c is -cov(fU,hU)/var(hU)
    -input:
        n : numbre of montecarlo iterations.
        f : function we want to calculate the integral of.
        h : function with high correlation to f
        meanh: integral of h between 0 and 1.
    -output:
        np.mean(fU)       : estimation of integral of f from 0 to 1.
        np.std(fU,ddof=1) : standard deviation of the estimation.
    """
    U = np.random.uniform(0,1,n)
    fU = f(U)
    hU = h(U)
    c = -np.cov(fU,hU)[0][1]/np.var(hU)
    ffU = fU+c*(hU-meanh)
    return np.mean(ffU),np.std(ffU,ddof=1)

def q(x):
    """
    -probability density function of the chosen distribution.
    -input:
        x    : scalar or numpy array if all operations used in q are vectorized
    -output:
        q(x) : scalar or numpy array depending on input.
    """
    return np.exp(x)/(np.e-1)

def Q_inv(x):
    """
    -inverse cumulative distribution function of the chosen distribution.
    -input:
        x    : scalar or numpy array if all operations used in Q_inv are vectorized
    -output:
        Q-1(x) : scalar or numpy array depending on input.
    """
    return np.log((np.e-1)*x+1)



def mc_importance_sampling(n,f,q,Q_inv):
    """
    -montecarlo estimators with importance sampling: instead of sampling from the uniform distribution, we choose a new distribution with probability density function q,that is easy to compute and close to being proportional to f.
    -input:
        n : numbre of montecarlo iterations.
        f : function we want to calculate the integral of.
        q : probability density function of new distribution
        Q_inv: inverse of cumulative distribution function of new distribution
    -output:
        np.mean(fU)       : estimation of integral of f from 0 to 1.
        np.std(fU,ddof=1) : standard deviation of the estimation.
    """
    U  = np.random.uniform(0,1,n)
    X  = Q_inv(U)
    fU = f(X)/q(X)
    return np.mean(fU),np.std(fU,ddof=1)



def generate_data(n):
    """
    -generating estimations and their standard deviation of all the montecarlo estimators cited above for different numbers of iterations.
    -input:
        n : maximum  number of montecarlo iterations. (estimations will be computed for growing number of iterations in range (10,n+10)
    -output:
        mc_iterations : array of iterations for the estimators
        mc_data_mean  : estimations of each montecarlo estimator, for different numbers of iterations
        mc_data_std   : standard deviation of each montecarlo estimator, for different numbers of iterations.
    """
    mc_iterations = []

    mc_data_mean = {}
    mc_data_std  = {}

    mc_data_mean["crude"]               = []
    mc_data_mean["antithetic"]          = []
    mc_data_mean["control_variate"]     = []
    mc_data_mean["importance_sampling"] = []

    mc_data_std["crude"]                = []
    mc_data_std["antithetic"]           = []
    mc_data_std["control_variate"]      = []
    mc_data_std["importance_sampling"]  = []

    for i in range(10,n+1,1):
        print(i,"/",n)
        mc_iterations.append(i)

        m,std =  mc_crude(n,f)
        mc_data_mean["crude"].append(m)
        mc_data_std["crude"].append(std)

        m,std =  mc_antithetic(n,f)
        mc_data_mean["antithetic"].append(m)
        mc_data_std["antithetic"].append(std)

        m,std =  mc_control_variate(n,f,h,4.0/3.0)
        mc_data_mean["control_variate"].append(m)
        mc_data_std["control_variate"].append(std)

        m,std =  mc_importance_sampling(n,f,q,Q_inv)
        mc_data_mean["importance_sampling"].append(m)
        mc_data_std["importance_sampling"].append(std)
    for key in mc_data_mean.keys():
        mc_data_mean[key] = np.array(mc_data_mean[key])
        mc_data_std[key] = np.array(mc_data_std[key])
    mc_iterations = np.array(mc_iterations)
    return mc_iterations,mc_data_mean,mc_data_std


def plot_data(mc_iterations,mc_data_mean,mc_data_std):
    """
    -plots data generated by method generate_data()
    -input:
        outputs of generate_data()
    -output:
        plot of estimations of the montecarlo estimators
        plot of standard deviation of montecarlo estimators
    """
    fig, axs = plt.subplots(2, 1)
    axs[0].axhline(y = 1.463, color = 'k',label='exact value')

    for key in mc_data_mean.keys():
        axs[0].plot(mc_iterations,mc_data_mean[key],label=key)
    axs[0].grid()
    axs[0].legend(loc=3,prop={'size':4})
    axs[0].set_title("convergence of MC methods")

    for key in mc_data_std.keys():
        axs[1].plot(mc_iterations,mc_data_std[key],label=key)
    axs[1].grid()
    axs[1].legend(loc=3,prop={'size':4})
    axs[1].set_title("standard deviation of MC methods")

    fig.tight_layout()



def plot_data_CI(mc_iterations,mc_data_mean,mc_data_std):
    """
    -plots data generated by method generate_data()
    -input:
        outputs of generate_data()
    -output:
        plot of estimations of the montecarlo estimators with 95% confidance intervales
    """
    plt.figure()
    colors = ['r','b','g','orange']
    c = 0
    plt.axhline(y = 1.463, color = 'k',label='exact value (1.463)')
    for key in mc_data_mean.keys():
        plt.plot(mc_iterations,mc_data_mean[key],label=key,color=colors[c])
        c = (c+1)%len(colors)

    c = 0
    for key in mc_data_std.keys():
        bound = 1.96*mc_data_std[key]/np.sqrt(mc_iterations)
        plt.plot(mc_iterations,mc_data_mean[key]+bound,color=colors[c],linestyle='dashed',linewidth=0.5)
        plt.plot(mc_iterations,mc_data_mean[key]-bound,color=colors[c],linestyle='dashed',linewidth=0.5)
        c = (c+1)%len(colors)
    plt.legend(loc=3,prop={'size':4})
    plt.title("convergence and 95% confidance interval")
    plt.grid()



def main():
    I,data_mean,data_std = generate_data(1000)
    plot_data(I,data_mean,data_std)
    plot_data_CI(I,data_mean,data_std)
    plt.show()







if __name__ == "__main__":
    main()



