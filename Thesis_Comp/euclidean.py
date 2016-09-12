#!/usr/bin/env python
# Leo Selker, 2016
from __future__ import division
import numpy as np
import numpy.linalg as linalg
from scipy.spatial import distance as dist
import math


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

        singular_values = linalg.svd(self.distance_matrix, compute_uv=False)
        if singular_values[-1] / singular_values[0] > self.EPSILON:  # Check for singular matrix
            mobius_inversion = linalg.inv(exp_distance_matrix)
            return np.sum(mobius_inversion)
        else:
            return None

    def multiscale_magnitude(self, scale_factor_iterable):
        magnitude_list = map(self.magnitude, scale_factor_iterable)
        return magnitude_list

if __name__ == '__main__':
    test_array = np.array(
        [[0, 0],
         [0, 2],
         [0, 4]]
    )
    euclidean_space = EuclideanSpace(test_array)
    print(euclidean_space.multiscale_magnitude((2, 3)))
