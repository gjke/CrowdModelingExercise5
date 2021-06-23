from numpy import linalg as LA

def shift_coordinates(x, m, delta_t):
    """
    shifts the values of x by k*delta_t, where k is 0, 1, 2, ..., m

    Parameters:
        model (VAE): VAE model
        n (int): number of samples to generate

    Returns:
        tuple: m array like objects of length len(x)-m*delta_t.

    """

    new_coordinates = tuple(
        [x[k*delta_t:len(x) if k == m else -(m-k)*delta_t]
         for k in range(1, m + 1)]
    )
    return (x[:-m*delta_t],) + new_coordinates

def get_v_from_data(x0,x1, t):
    """
    Returns a velocity vector for the two given sets of points x0 and x1

    Parameters:
        x0: Start position
        x1: End position
        t : set the time it took to reach the end position

    Returns:
        tuple: the velocity necessary for the point to reach its target in the given time

    """
    return (x1-x0)/t

def mse(x0,x1):
    """
    Returns the mean square error for the given points

    Parameters:
        x0: Set of points
        x1: Set of points

    Returns:
        float: mean square error of all the points

    """
    N = x0.shape[0]
    s = 0
    for i in range(N):
        distance = LA.norm(x0[i]-x1[i],ord = 2)**2
        s = s + distance
    return s/N