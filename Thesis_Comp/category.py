#!/usr/bin/env python
# Leo Selker, 2016
from __future__ import division
import numpy as np
import numpy.linalg as linalg


class PartialCategory(object):
    # Objects and # of morphisms. Does not keep track of or enforce composition.
    EPSILON = 10**(-8)  # No real reason...

    def __init__(self,  morphism_table):
        """
        :param morphism_table: nxn numpy array of # morphisms
        """
        self.morphism_table = morphism_table
        (a, b) = morphism_table.shape
        if a != b:
            raise ValueError("Morphism table must be square")

    def mobius_matrix(self):
        singular_values = linalg.svd(self.morphism_table, compute_uv=False)
        if singular_values[-1] / singular_values[0] > self.EPSILON:  # Check for singular matrix
            return linalg.inv(self.morphism_table)
        else:
            return None

    def weights(self):
        mobius_matrix = self.mobius_matrix()
        if mobius_matrix == None: return None
        weights = []
        for column in range(mobius_matrix.shape[1]):
            weight = np.sum(mobius_matrix[:, column])
            weights.append(weight)
        return weights

if __name__ == '__main__':
    morphism_tables = (
        np.array(
            [[1,0,0,0,1,1],
             [0,1,0,1,0,1],
             [0,0,1,1,1,0],
             [0,0,0,1,0,0],
             [0,0,0,0,1,0],
             [0,0,0,0,0,1]]
        ),
        np.array(
            [[1,1,1],
             [0,1,1],
             [0,0,1]]
        ),
        np.array(
            [[1, 0, 0, 0, 1, 1, 0],
             [0, 1, 0, 1, 0, 1, 0],
             [0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 1],
             [0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1]]
        )
    )

    p_cat = PartialCategory(morphism_tables[2])
    print(p_cat.mobius_matrix(), np.sum(p_cat.mobius_matrix()))