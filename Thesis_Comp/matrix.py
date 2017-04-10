from matplotlib import pyplot as plt
import numpy as np
import math 

def spec_norm(A):
    svd = np.linalg.svd(A)
    return svd[1][0]

def calc_E(d, scale=1):
    epsilon = math.exp(-scale*(1 + d)) - math.exp(-scale*math.sqrt(1 + d**2))
    E = np.zeros((4,4))
    Em = np.matrix(E)
    for (i,j) in [(0,3), (1,2), (2,1), (3,0)]:
        Em[i,j] = epsilon
    return Em

def inv_with_error(d, scale=1, n_terms=3):
    """ 
    Returns a tuple
    (A^1, (A - E)^-1, Magnitude Error Bound)
    """
    # Put together A
    A = np.empty((4,4))
    for i in range(4):
        A[i,i] = 1
    for (i,j) in [(0,1), (1,0), (2,3), (3,2)]:
        A[i,j] = math.exp(-scale)
    for (i,j) in [(0,2), (2,0), (1,3), (3,1)]:
        A[i,j] = math.exp(-scale*d)
    for (i,j) in [(0,3), (1,2), (2,1), (3,0)]:
        A[i,j] = math.exp(-scale*(1 + d))
    A = np.matrix(A)
    E = calc_E(d, scale)
    ET = 0
    for i in range(1, n_terms):
        ET = ET + mp((E * (inv(A))),i)
    eb = abs(np.sum(ET))
    inv_A = inv(A)
    inv_AE = inv(A - E)
    # actual = np.sum(inv_A - inv_AE)
    return (inv_A, inv_AE, eb)
        
def magnitudes_error(d, scale=1, n_terms=3):
    # returns (mag(A), mag(A-E), eb)
    (Ainv, AEInv, eb) = inv_with_error(d, scale, n_terms)
    return (np.sum(Ainv), np.sum(AEInv), eb)

def mp(A,n):
    return np.linalg.matrix_power(A,n)

def inv(A):
    return np.linalg.inv(A)

def prod_approx_plots():
    d = 1
    scale_factors_low = np.arange(.1, 5, .1)
    scale_factors_mid = np.arange(5, 1000, .5)
    scale_factors_high = np.arange(1000, 50000, 100)
    scale_factors = np.concatenate((scale_factors_low, scale_factors_mid, scale_factors_high))
    

    scale_factors_small = scale_factors * (.001)
    scale_factors_log = map(lambda x: math.log(x, 10), scale_factors)
    product_magnitudes = map(
        lambda x: magnitudes_error(d,scale=x)[0], scale_factors)
    euc_magnitudes = map(
        lambda x: magnitudes_error(d,scale=x)[1], scale_factors)
    errors = map(
        lambda x: magnitudes_error(d,scale=x)[2], scale_factors)
    
    plt.title("Product Space Magnitude")
    plt.xlabel("Scale factor, log scale")
    plt.ylabel("Magnitude")
    #plt.plot(scale_factors_log, product_magnitudes, 'r', 
    #         label='Product Space') 
    plt.plot(scale_factors_log, euc_magnitudes, 'r',
             label='Euclidean Approximation')
    plt.errorbar(scale_factors_log, product_magnitudes, yerr=errors,fmt='b',
                 label='Product Space Magnitude, Error Bounds')
    font = {'family' : 'normal',
            'size'   : 24}

    plt.rc('font', **font)
    plt.axis([-1.0,1.2,1,4.1])
    plt.legend(loc='ul')
    plt.show()

