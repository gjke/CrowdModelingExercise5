from scipy.spatial import distance_matrix
import numpy as np
from numpy import linalg as LA


class RBF:

    def __init__(self, initial_data, transformed_data, amount_of_centers, epsilon, delta_t):
        """

        :type initial_data: object
        :type transformed_data: object
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
        F = self.get_approximate_velocity(initial_data, transformed_data)
        C, residuals, rank, s = LA.lstsq(theta, F, 1000)
        self.C = C

    def get_approximate_velocity(self, x0, x1):
        return (x1 - x0) / self.delta_t

    def rbf(self, to_predict):
        distance = distance_matrix(
            to_predict, self.centers)
        theta = np.exp(-((distance) / (self.epsilon))**2)

        return (theta @ self.C)
