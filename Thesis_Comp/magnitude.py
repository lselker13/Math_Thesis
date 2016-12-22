import numpy as np

class EuclideanSpace(object):

    def __init__(self, data_array):
        self.data_array = data_array
        (self.n_points, self.dimension) = data_array.shape


    def compute_distance_matrix(self):
        # Computes the distance matrix for the space
