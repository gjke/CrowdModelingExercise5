from scipy.spatial import distance_matrix
import numpy as np
from numpy import linalg as LA


class RBF:

    def __init__(self, initial_data, transformed_data, amount_of_centers, epsilon, delta_t, vector_field=True):
         """
        Constructor for the RBF class

        Parameters:
            initial_data: The initial data
            transformed_data: The data updated by an update function that we want to approximate
            amount_of_centers: how many radial basis functions the approximated function should use
            epsilon: how wide the radial basis functions are supposed to be
            delta_t: the time it took to get from the initial_data to the transformed_data
            vector_field: if true then the calculations are done for a vector field and the approximated function will give a velocity and not predicted points
        """
        if transformed_data.ndim == 1:
            self.transformed_data = transformed_data.reshape(-1, 1)
        else:
            self.transformed_data = transformed_data

        if initial_data.ndim == 1:
            self.initial_data = initial_data.reshape(-1, 1)
        else:
            self.initial_data = initial_data
        self.amount_of_centers = amount_of_centers
        self.epsilon = epsilon
        self.delta_t = delta_t
        self.centers = self.initial_data[np.random.choice(
            self.initial_data.shape[0], size=self.amount_of_centers)]
        distance = distance_matrix(
            self.initial_data, self.centers)
        theta = np.exp(-(distance / epsilon)**2)
        F = self.get_approximate_velocity(
            initial_data, transformed_data) if vector_field else self.transformed_data
        C, residuals, rank, s = LA.lstsq(theta, F, 1000)
        self.C = C

    def get_approximate_velocity(self, x0, x1):
        """
        Returns a velocity vector for the two given sets of points x0 and x1

        Parameters:
            x0: Start position
            x1: End position

        Returns:
            tuple: the velocity necessary for the point to reach its target in the time delta_t

        """
        return (x1 - x0) / self.delta_t

    def rbf(self, to_predict):
        """
        uses the approximated function on the given points

        Parameters:
            to_predict: the points for which the approximated function is going to be calculated

        Returns:
            array of points: The outcome of the calculation

        """
        distance = distance_matrix(
            to_predict, self.centers)
        theta = np.exp(-((distance) / (self.epsilon))**2)

        return (theta @ self.C)
