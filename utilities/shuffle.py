import numpy as np

def shuffle_fr_random(array, num_surrogates):
    """
    Generates a list of shuffled (surrogate) arrays.

    Parameters:
    - array: The 1D array to be shuffled.
    - num_surrogates: Number of surrogate arrays to generate.

    Returns:
    - surrogates: A list of surrogate arrays.
    """
    
    if array.ndim != 1:
        raise ValueError("The input array should be 1-dimensional.")
    
    shuffles = [np.random.permutation(array) for _ in range(num_surrogates)]
    
    return shuffles