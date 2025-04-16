"""Scripts containing helper functions for working with arrays."""

import warnings
import numpy as np
from collections.abc import Iterable


def validate_nd_arrays(arr1, arr2):
    """Validates that two n-dimensional arrays have the same shape and values.

    Parameters
    ----------
    arr1 : np.ndarray
        First n-dimensional array.
    arr2 : np.ndarray
        Second n-dimensional array.
    """
    if arr1.ndim != arr2.ndim:
        raise ValueError("Arrays have different number of dimensions.")
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays have different shapes.")
    if not np.equal(arr1, arr2).all():
        raise ValueError("Values in the arrays do not match.")


def flatten_nested_data(nested_structure):
    """Flattens an arbitrarily nested structure (containing lists, tuples, numpy arrays,
    or other items) into a single, flat 1D NumPy array.

    This function iterates through the nested structure.
    - If it encounters an iterable (like list or tuple, but not str, bytes, or np.ndarray),
      it recurses deeper.
    - If it encounters a NumPy array, it flattens that array and yields its elements.
    - Otherwise, it yields the item directly (assuming it's a base element like int, float, str).

    Parameters
    ----------
    nested_structure : nested list, tuple, or other iterable
        The arbitrarily nested dataset.

    Returns
    --------
    array : np.ndarray
        A 1D numpy array containing all non-iterable elements and all elements from flattened
        numpy arrays.
        Returns an empty array if the input is empty or contains only empty structures.
        The dtype is inferred by NumPy, but may become 'object' if incompatible types are mixed.
    """

    def _yield_elements(item):
        """Recursive generator helper function."""
        # Check if the item is an iterable we should recurse into.
        if isinstance(item, Iterable) and not isinstance(
            item, (str, bytes, np.ndarray)
        ):
            for sub_item in item:
                yield from _yield_elements(sub_item)
        elif isinstance(item, np.ndarray):
            yield from item.flatten()
        else:
            yield item

    # Collect all elements into a list
    elements_list = list(_yield_elements(nested_structure))

    # Convert the final list of elements to a numpy array
    if not elements_list:
        return np.array([])

    try:
        return np.array(elements_list)
    except ValueError:
        # If standard conversion fails, create an array with dtype=object
        warnings.warn(
            "Warning: Could not create a uniform numeric array due to mixed types. "
            "Returning array with dtype=object."
        )
        return np.array(elements_list, dtype=object)
