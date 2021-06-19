from scipy.spatial import distance_matrix
import numpy as np
from numpy import linalg as LA

class RBF:

    def __init__(self, initial_data,transformed_data,amount_of_centers,epsilon,delta_t):
        """

        :type initial_data: object
        :type transformed_data: object
        """
        self.transformed_data = transformed_data
        self.initial_data = initial_data
        self.amount_of_centers = amount_of_centers
        self.epsilon = epsilon
        self.delta_t = delta_t
        distance = distance_matrix(initial_data, initial_data[:amount_of_centers])
        theta = np.exp(-(distance / epsilon)**2)
        F = self.get_approximate_velocity(initial_data,transformed_data)
        C, residuals, rank, s = LA.lstsq(theta, F, 1000)
        self.C = C


    def get_approximate_velocity(self, x0, x1):
        return (x1 - x0) / self.delta_t

    def rbf(self,to_predict):
        distance = distance_matrix(to_predict,self.initial_data[:self.amount_of_centers])
        theta = np.exp(-((distance) / (self.epsilon))**2)

        return (theta @ self.C)