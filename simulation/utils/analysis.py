"""Functions for post-hoc data analysis."""

import warnings
import numpy as np
from scipy.spatial.distance import jensenshannon
from osl_dynamics.inference import metrics


def calculate_rv_coefficients(covs1, covs2):
    """Computes the RV coefficients between two sets of covariance matrices.

    Parameters
    ----------
    covs1 : np.ndarray
        First set of covariance matrices.
        Shape must be (n_states, n_channels, n_channels).
    covs2 : np.ndarray
        Second set of covariance matrices.
        Shape must be (n_states, n_channels, n_channels).

    Returns
    -------
    rv_coeffs : list
        List of RV coefficients for each pair of covariance matrices.
        Shape is (n_states,).
    """
    # Validate inputs
    if covs1.shape != covs2.shape:
        raise ValueError("Covariance matrices have different shapes.")

    return [
        metrics.pairwise_rv_coefficient(np.array([c1, c2]))[0, 1]
        for c1, c2 in zip(covs1, covs2)
    ]


def calculate_js_distance(data1, data2, bounds=None, n_bins=12, base=2):
    """Computes the Jensen-Shannon distance between two datasets.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.
    data2 : np.ndarray
        Second data array.
    bounds : list
        Bounds for the histogram bins.
        If None, the min and max of both datasets are used.
    n_bins : int
        Number of bins for the histogram. Default is 10.
    base : int
        Base for the logarithm used in the Jensen-Shannon distance calculation. Default is 2.
        If base is 2, the distance is normalized to [0, 1].
        If base is np.e, the distance is normalized to [0, log(2)].

    Returns
    -------
    jsd : float
        Jensen-Shannon distance between the two data arrays.
    """
    # Validate inputs
    if not (isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray)):
        raise ValueError("Both inputs must be numpy arrays.")

    if bounds is None:
        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
    else:
        min_val, max_val = bounds[0], bounds[1]

    if min_val >= max_val:
        # Handle cases where all data points are identical or range is zero
        if np.allclose(data1, data2):
            return 0.0  # distributions are identical if data is identical
        else:
            raise ValueError("Invalid bounds: min_val must be less than max_val.")

    # Define common bin edges
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    # Compute histograms (counts) using the common bins
    counts1, _ = np.histogram(data1, bins=bin_edges, density=False)
    counts2, _ = np.histogram(data2, bins=bin_edges, density=False)

    # Handle cases where histograms might be empty (e.g., if bins don't capture data)
    if counts1.sum() == 0 or counts2.sum() == 0:
        # This might happen if data falls exactly on bin edges inconsistently or
        # other edge cases.
        warnings.warn("Histogram counts are zero for one or both datasets.")
        return np.nan  # indicates an issue

    # Normalize counts to get empirical probability distributions
    p = counts1 / counts1.sum()
    q = counts2 / counts2.sum()

    # Compute the Jensen-Shannon distance
    jsd = jensenshannon(p, q, base=base)

    return jsd
