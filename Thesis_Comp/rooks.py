#!/usr/bin/env python
from __future__ import division
import math
import numpy as np
"""
Generate the min(p,q) x min(p,q) matrix to get g_k in terms of b_k
"""

def chars(m,n):
    c = [[round(char(i,j)) for i in range(1,m)] for j in range(1,n)]
    return np.round(np.array(c))


#TODO: Support floats/complex numbers 
def char(p,q):
    A = np.abs(generate(p,q))
    B = np.linalg.inv(A)
    return np.round(np.sum(B),decimals=0)

def generate(p,q):
    # Packages the coefficients into matrix
    n = min(p,q)

    l = [
        [coef(p,q,k,i) for i in range(n+1)]
        for k in range(n+1)]

    return np.array(l)

def coef(p,q,k,i):
    # Returns the k,i entry in the matrix for a pxq board
    ordered = ((-1)**i * ch(k,i) * math.factorial(i) 
               * falling(p-i, k-i) * falling(q-i, k-i))

    return ordered // math.factorial(k)
                

def coef_(p,q,k,i):
    # Returns the k,i entry in the matrix for a pxq board
    ordered = ((-1)**(i+k) * ch(k,i) * math.factorial(i) 
               * falling(p-i, k-i) * falling(q-i, k-i))

    return ordered // math.factorial(k)
                

def ch(a,b):
    # Choose
    if a < b: return 0
    else: return math.factorial(a) // (math.factorial(b)*math.factorial(a-b))

def falling(a,b):
    # Falling factorial
    if a < b: return 0
    total = 1
    offset = a - b + 1
    for i in range(b):
        total *= (i + offset)
    return total
