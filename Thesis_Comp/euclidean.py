#!/usr/bin/env python
# Leo Selker, 2016
from __future__ import division
import numpy as np
from scipy.spatial import distance as dist
import math
from matplotlib import pyplot as plt
import category as cat

class EuclideanSpace(object):
    EPSILON = 10**(-8)  # No real reason...

    def __init__(self, data_array, scale_factor=1):
        # Data should be row major - each row is a point. Should be a numpy array.
        # We multiply early by the scale factor to give the option of permanently scaling the space
        data_array *= scale_factor
        (self.n_points, self.dimension) = data_array.shape

        # Calculate the distance matrix (using scipy)
        distance_matrix_compact = dist.pdist(data_array, 'euclidean')
        self.distance_matrix = dist.squareform(distance_matrix_compact)

    def magnitude(self, scale_factor=1):

        # Scale then exponentiate the distances
        vector_inverse_exp = np.vectorize(lambda n: math.exp(-n))
        exp_distance_matrix = vector_inverse_exp(self.distance_matrix * scale_factor)

        cat_rep = cat.PartialEnrichedCategory(exp_distance_matrix)
        return np.sum(cat_rep.mobius_matrix())

    def multiscale_magnitude(self, scale_factor_iterable):
        magnitude_list = map(self.magnitude, scale_factor_iterable)
        return magnitude_list

if __name__ == '__main__':
    test_arrays = (
        np.array(
            [[0, 0],
             [0, .01],
             [10, 0],
             [10,.01]]
        ),
        np.array(
            [[0, 0],
            [10, 0]]
        )
    )
    euclidean_space = EuclideanSpace(test_arrays[0])
    two_space = EuclideanSpace(test_arrays[1])
    scale_factors = np.arange(.05, 20, .05)
    magnitudes_euc = euclidean_space.multiscale_magnitude(scale_factors)
    magnitudes_two = two_space.multiscale_magnitude(scale_factors)
    scale_factors_log = map(lambda x: math.log(x, 10), scale_factors)

    #plt.subplot(211)
    #plt.plot(scale_factors_log, magnitudes_euc)

    #plt.subplot(212)
    plt.plot(scale_factors_log, magnitudes_two)

    plt.show()


