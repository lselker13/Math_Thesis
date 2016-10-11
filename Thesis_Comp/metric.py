#!/usr/bin/env python
# Leo Selker, 2016
from __future__ import division
import numpy as np
from scipy.spatial import distance as dist
import math
from matplotlib import pyplot as plt
import category as cat

EPSILON = 10**(-5)

class MetricSpace(object):
    EPSILON = 10**(-8)  # No real reason...
    distance_matrix = None

    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

    @staticmethod
    def space(distance_matrix):
        return MetricSpace(distance_matrix)

    @staticmethod
    def product_space(matrix_1, matrix_2):
        # Slow
        ((m, n), (p,q)) = (matrix_1.shape, matrix_2.shape)
        distance_matrix = np.empty((m*p, n*q))
        for r, row in enumerate(matrix_1):
            for s, el in enumerate(row):
                distance_matrix[p*r: p*(r+1), q*s: q*(s+1)] = matrix_2 + el
        return MetricSpace(distance_matrix)

    @staticmethod
    def euclidean(data_array, scale_factor=1):
        # Builds a metric space from euclidean coordinates
        # Data should be row major - each row is a point. Should be a numpy array.
        # We multiply early by the scale factor to give the option of permanently scaling the space
        data_array *= scale_factor

        # Calculate the distance matrix (using scipy)
        distance_matrix_compact = dist.pdist(data_array, 'euclidean')
        distance_matrix = dist.squareform(distance_matrix_compact)
        return MetricSpace(distance_matrix)

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
    if False: # Euclidean tests for 2- and 4-point spaces
        test_arrays_euclidean = (
            np.array(
                [[0, 0],
                 [0, .001],
                 [1, 0],
                 [1,.001]]
            ),
            np.array(
                [[0, 0],
                [1, 0]]
            ),
        )
        # test_distance_matrices = (
        #     np.array(
        #         [0,10]
        #     ),
        # )
        scale_factors = np.arange(.5, 10000, .5)
        scale_factors_log = map(lambda x: math.log(x, 10), scale_factors)

        euclidean_space = MetricSpace.euclidean(test_arrays_euclidean[0])
        two_space = MetricSpace.euclidean(test_arrays_euclidean[1])
        magnitudes_euc = euclidean_space.multiscale_magnitude(scale_factors)
        magnitudes_two = two_space.multiscale_magnitude(scale_factors)

        #plt.subplot(211)
        plt.title("Four Point Space Magnitude")
        plt.xlabel("Scale factor, log scale")
        plt.ylabel("Magnitude")
        plt.plot(scale_factors_log, magnitudes_euc)

        # plt.subplot(212)
        # plt.plot(scale_factors_log, magnitudes_two)

        plt.show()
    if True:
        scale_factors = np.arange(.5, 10000, .5)

        scale_factors_log = map(lambda x: math.log(x, 10), scale_factors)

        matrix_1 = np.array(
            [[0, 1, 2], [1, 0, 2], [2, 2, 0]]
        )
        matrix_2 = np.array(
            [[0, .001], [.001, 0]]
        )

        space_1 = MetricSpace.space(matrix_1)
        space_2 = MetricSpace.space(matrix_2)
        product_space = MetricSpace.product_space(matrix_1, matrix_2)

        magnitudes_1 = space_1.multiscale_magnitude(scale_factors)
        magnitudes_2 = space_2.multiscale_magnitude(scale_factors)
        product_of_magnitudes = [el1 * el2 for el1, el2 in zip(magnitudes_1, magnitudes_2)]

        magnitudes_product = product_space.multiscale_magnitude(scale_factors)

        for i, (el1, el2) in enumerate(zip(product_of_magnitudes, magnitudes_product)):
            assert abs(el1 - el2) < EPSILON, "Arrays don't match at element {}".format(i)

        plt.subplot(211)
        plt.plot(scale_factors_log, product_of_magnitudes)

        plt.subplot(212)
        plt.plot(scale_factors_log, magnitudes_product)
        plt.show()