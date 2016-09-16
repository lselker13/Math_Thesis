#!/usr/bin/env python
# Leo Selker, 2016

from __future__ import division
import numpy as np
import category as cat

class SimplicialComplex(object):
    # NOT as fast/good as it could be
    def __init__(self, simplex_list):
        """
        :param maximal_simplex_list: A list of sets.
        """
        self.simplex_list = simplex_list
        self.simplex_list.sort(key=lambda l:len(l))

    def to_partial_category(self):
        """

        :return: A partial category corresponding to the simplicial complex.
        """
        n_simplices = len(self.simplex_list)
        morphism_table = np.zeros((n_simplices, n_simplices))
        for i, simplex in enumerate(self.simplex_list):
            morphism_table[i,i] = 1
            for j, other in enumerate(self.simplex_list[:i]):
                if other.issubset(simplex):
                    morphism_table[j, i] = 1
        return cat.PartialCategory(morphism_table)

if __name__ == '__main__':
    simplex_list = [{1},{2},{3},{1,2},{2,3},{3,1}, {1,2,3}]
    # simplex_list = [{1},{2},{3},{4},{1,2},{2,3},{3,4},{4,1},{2,4},{1,2,4}]
    simplicial_complex = SimplicialComplex(simplex_list)
    print simplicial_complex.simplex_list
    partial_category = simplicial_complex.to_partial_category()
    print partial_category.morphism_table
    print partial_category.mobius_matrix()