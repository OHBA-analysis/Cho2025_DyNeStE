"""Helper functions for working with arrays."""

import numpy as np
from decimal import Decimal, ROUND_HALF_UP


def round_nonzero_decimal(num, precision=1, method="round"):
    """
    Rounds an input decimal number starting from its first non-zero value.

    For instance, with precision of 1, we have:
    0.09324 -> 0.09
    0.00246 -> 0.002

    Parameters
    ----------
    num : float
        Float number.
    precision : int
        Number of decimals to keep. Defaults to 1.
    method : str
        Method for rounding a number. Currently supports
        np.round(), np.floor(), and np.ceil().

    Returns
    -------
    round_num : float
        Rounded number.
    """
    # Validation
    if num > 1:
        raise ValueError("num should be less than 1.")
    if num == 0: return 0
    
    # Identify the number of zero decimals
    decimals = int(abs(np.floor(np.log10(abs(num)))))
    precision = decimals + precision - 1
    
    # Round decimal number
    if method == "round":
        round_num = np.round(num, precision)
    elif method == "floor":
        round_num = np.true_divide(np.floor(num * 10 ** precision), 10 ** precision)
    elif method == "ceil":
        round_num = np.true_divide(np.ceil(num * 10 ** precision), 10 ** precision)
    
    return round_num


def round_up_half(num, decimals=0):
    """
    Rounds a number using a 'round half up' rule. This function always
    round up the half-way values of a number.

    NOTE: This function is added because Python's default round() 
    and NumPy's np.round() functions use 'round half to even' method.
    Their implementations mitigate positive/negative bias and bias 
    toward/away from zero, while this function does not. Hence, this 
    function should be preferentially only used for the visualization 
    purposes.

    Parameters
    ----------
    num : float
        Float number.
    decimals : int
        Number of decimals to keep. Defaults to 0.

    Returns
    -------
    round_num : float
        Rounded number.
    """
    multiplier = 10 ** decimals
    round_num = float(
        Decimal(num * multiplier).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / multiplier
    )
    return round_num


def window_shuffle(array, shuffle_window_length, rng):
    """
    Shuffles the input array in windows of specified length.

    Parameters
    ----------
    array : np.ndarray or list of np.ndarray
        Input array or list of arrays to be shuffled.
    shuffle_window_length : int
        Length of the window for shuffling.
    rng : np.random.Generator
        Random number generator to use for shuffling. If None, a new
        random number generator will be created.

    Returns
    -------
    shuffled_array : np.ndarray or list of np.ndarray
        Shuffled array or list of arrays.
    """
    # Validate input
    if isinstance(array, np.ndarray):
        array = [array]

    # Set random seed
    if rng is not None:
        rng = np.random.default_rng()
    
    # Shuffle the array in windows
    shuffled_array = []
    for arr in array:
        # Trim the array to fit the window size
        n_samples = arr.shape[0]
        n_windows = n_samples // shuffle_window_length
        arr = arr[:n_windows * shuffle_window_length]

        # Reshape the array into windows
        arr_reshaped = arr.reshape(
            n_windows, shuffle_window_length, *arr.shape[1:]
        )

        # Permute the window order
        perm = rng.permutation(n_windows)
        arr_shuffled = arr_reshaped[perm]

        # Reshape back to the original shape
        arr_shuffled = arr_shuffled.reshape(
            -1, *arr.shape[1:]
        )
        
        shuffled_array.append(arr_shuffled)

    return shuffled_array[0] if len(shuffled_array) == 1 else shuffled_array
