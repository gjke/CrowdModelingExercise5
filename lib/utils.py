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
