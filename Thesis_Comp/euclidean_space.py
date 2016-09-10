from __future__ import division
import numpy as np
import numpy.linalg as linalg
import sys
from scipy.spatial import distance as dist
import math


class EuclideanSpace(object):
    EPSILON = 10**8  # No real reason...

    def __init__(self, data_array, scale_factor=1):
        # Data should be row major - each row is a point. Should be a numpy array.
        data_array = data_array * scale_factor
        print data_array
        (self.n_points, self.dimension) = data_array.shape

        # Calculate the distance matrix (using scipy)
        distance_matrix_compact = dist.pdist(data_array, 'euclidean')
        distance_matrix = dist.squareform(distance_matrix_compact)
        # Go through and exponentiate the distances
        vector_inverse_exp = np.vectorize(lambda n: math.exp(-n))
        self.distance_matrix = vector_inverse_exp(distance_matrix)


    def magnitude(self):
        if linalg.cond(self.distance_matrix) < 1 / self.EPSILON or True:
            mobius_inversion = linalg.inv(self.distance_matrix)
            return np.sum(mobius_inversion)
        else:
            return None

if __name__ == '__main__':
    test_array = np.array(
        [[0, 0],
         [0, 2],
         [3, 2]]
    )
    print EuclideanSpace(test_array, scale_factor=10).magnitude()
