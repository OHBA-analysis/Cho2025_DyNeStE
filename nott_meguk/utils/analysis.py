"""Functions for pos-hoc analysis."""

import numpy as np
from sklearn.metrics import mutual_info_score
from tqdm import trange
from osl_dynamics.inference import metrics, modes


def calculate_summary_stats(stc, sampling_frequency):
    """Calculates summary statistics of state time courses.

    Parameters
    ----------
    stc : list of np.ndarray
        List of state time courses for each subject.
        Shape is (n_subjects, n_samples, n_states).
    sampling_frequency : int
        Sampling frequency of the data.

    Returns
    -------
    fo : np.ndarray
        Fractional occupancies of the states. Shape is (n_subjects, n_states).
    lt : np.ndarray
        Mean lifetimes of the states. Shape is (n_subjects, n_states).
    intv : np.ndarray
        Mean intervals of the states. Shape is (n_subjects, n_states).
    sr : np.ndarray
        Switching rates of the states. Shape is (n_subjects, n_states).
    """
    # Compute summary statistics
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(
        stc, sampling_frequency=sampling_frequency
    )
    intv = modes.mean_intervals(
        stc, sampling_frequency=sampling_frequency
    )
    sr = modes.switching_rates(
        stc, sampling_frequency=sampling_frequency
    )
    return fo, lt, intv, sr


def compute_statewise_correlation(data_1, data_2):
    """Calculates the correlation between given time series.

    Parameters
    ----------
    data_1 : np.ndarray
        First time series. Shape must be (n_samples, n_states).
    data_2 : np.ndarray
        Second time series. Shape must be (n_samples, n_states).

    Returns
    -------
    corr : np.ndarray
        Correlation between each state in the corresponding alphas.
        Shape is (n_states, n_states).
    """
    # Validation
    if data_1.shape[1] != data_2.shape[1]:
        raise ValueError(
            "data_1 and data_2 shapes are incomptible. "
            + f"(data_1.shape={data_1.shape}, data_2.shape={data_2.shape})"
        )
    n_states = data_1.shape[1]

    # Adjust the length of two data
    n_keep = min(data_1.shape[0], data_2.shape[0])
    data_1 = data_1[:n_keep, :]
    data_2 = data_2[:n_keep, :]
    # NOTE: If different number of time-delay embeddings or sliding window was used during data preparation, 
    #       this adjustment is not sufficient, and the correlation is likely to be flawed. This adjustment 
    #       is only for adjusting different sequence lengths.

    # Compute correlation
    corr = np.corrcoef(data_1, data_2, rowvar=False)
    corr = corr[:n_states, n_states:]
    
    return corr


def compute_statewise_riemannian_distance(cov_1, cov_2):
    """Calculates the Riemannian distance between two state-wise 
       covariance matrices.

    Parameters
    ----------
    cov_1 : np.ndarray
        First covariance matrix.
        Shape must be (n_states, n_channels, n_channels).
    cov_2 : np.ndarray
        Second covariance matrix.
        Shape must be (n_states, n_channels, n_channels).

    Returns
    -------
    riemannian_distances : np.ndarray
        Riemannian distances between covariance matrices of the
        corresponding states. Shape is (n_states, n_states).
    """
    # Validation
    if cov_1.shape[0] != cov_2.shape[0]:
        raise ValueError(
            "cov_1 and cov_2 shapes are incomptible. "
            + f"(cov_1.shape={cov_1.shape}, cov_2.shape={cov_2.shape})"
        )
    if cov_1.ndim != 3:
        raise ValueError("cov_1 and cov_2 must be 3D arrays.")

    # Ensure data types
    cov_1 = cov_1.astype(np.float64)
    cov_2 = cov_2.astype(np.float64)

    # Compute Riemannian distances
    n_states = cov_1.shape[0]
    riemannian_distances = np.zeros((n_states, n_states))
    for i in trange(n_states, desc="Computing Riemannian distances"):
        for j in range(i, n_states):
            riemannian_distances[i, j] = metrics.riemannian_distance(cov_1[i], cov_2[j])

    # Fill the lower triangle to make the matrix symmetric
    diagonal_values = np.diag(riemannian_distances)  # to prevent double counting
    riemannian_distances += (riemannian_distances.T - np.diag(diagonal_values))
    
    return riemannian_distances


def compute_dice_coefficients(stc_1, stc_2):
    """Calculates the dice coefficients between two state time courses.

    Parameters
    ----------
    stc_1 : list of np.ndarray
        First state time course for each subject.
        Shape is (n_subjects, n_samples, n_states).
    stc_2 : list of np.ndarray
        Second state time course for each subject.
        Shape is (n_subjects, n_samples, n_states).

    Returns
    -------
    dice : np.ndarray
        Dice coefficients for each subject.
        Shape is (n_subjects,).
    """
    # Validation
    if len(stc_1) != len(stc_2):
        raise ValueError(
            "stc_1 and stc_2 must have the same number of subjects."
        )
    
    # Get data dimensions
    n_subjects = len(stc_1)

    # Compute dice coefficients for each subject
    dice = np.zeros((n_subjects,))
    for n in range(n_subjects):
        if stc_1[n].shape != stc_2[n].shape:
            raise ValueError(
                f"stc_1 and stc_2 must have the same shape for subject {n + 1}."
            )
        dice[n] = metrics.dice_coefficient(stc_1[n], stc_2[n])
        
    return dice


def compute_entropy(array, unit="nats"):
    """Computes the Shannon entropy of a discrete 1D array.
    
    Parameters
    ----------
    array : np.ndarray
        1D input array of discrete values.
    unit : str, optional
        The unit of entropy to return. 
        Options are "nats" (default) or "bits".
    
    Returns
    -------
    entropy : float
        The Shannon entropy of the input array.
    """
    # Validation
    if array.ndim != 1:
        raise ValueError("Input array must be 1D.")
    if unit not in ["nats", "bits"]:
        raise ValueError("Unit must be 'nats' or 'bits'.")
    
    # Count occurrences of each unique value
    counts = np.bincount(array)
    probs = counts / len(array)

    # Calculate entropy
    if unit == "nats":
        log = np.log
    elif unit == "bits":
        log = np.log2
    nonzero_idx = probs > 0  # only consider non-zero probabilities to avoid log(0)
    entropy = -np.sum(probs[nonzero_idx] * log(probs[nonzero_idx]))
    # NOTE: The entropy is not normalized, so the maximum value is not 1.
    #       The maximum value is log(n_discrete), where n_discrete is the
    #       number of unique discrete values.
    return entropy
    

def compute_mutual_information(
    state_time_course,
    lags,
    normalize=True,
    exclude_self=True,
    unit="nats",
):
    """Computes the mutual information between state time courses.
    
    Parameters
    ----------
    state_time_course : list of np.ndarray
        List of state time courses for each subject.
        Shape is (n_subjects, n_samples, n_states).
    lags : list or np.ndarray
        List of lags to compute mutual information for.
        Each lag is an integer representing the time delay.
    normalize : bool, optional
        Whether to normalize the mutual information by the geometric
        mean of the entropies. Defaults to True.
    exclude_self : bool, optional
        Whether to exclude self-information (i.e., mutual information
        between the same state). Defaults to True.
    unit : str, optional
        The unit of entropy to use for normalization. 
        Options are "nats" (default) or "bits".

    Returns
    -------
    avg_mi : np.ndarray
        State-averaged mutual information across subjects.
        Shape is (n_subjects, n_lags, n_states).
    mi_metric : np.ndarray
        Mutual information metric for each subject.
        Shape is (n_subjects, n_lags, n_states, n_states).
    """  
    # Get data dimensions
    n_subjects = len(state_time_course)
    _, n_states = state_time_course[0].shape
    n_lags = len(lags)

    # Trim subject-wise state time courses to the same length
    n_samples = min(stc.shape[0] for stc in state_time_course)
    for n in range(n_subjects):
        state_time_course[n] = state_time_course[n][:n_samples]
    # NOTE: Using per-subject n_samples can lead to more stable estiamtes for the longer 
    #       time courses and introduce higher variance in MI for the shorter time courses.

    # Initialize MI metric matrix
    mi_metric = np.zeros((n_subjects, n_lags, n_states, n_states))

    # Compute mutual information (only lower trainagle, i <= j)
    print(f"Computing MI for {n_states} states at lags {lags} ...")
    for n in trange(n_subjects):
        for i in range(n_states):
            for j in range(i, n_states):
                for idx, lag in enumerate(lags):
                    if lag < 0:
                        valid_range = slice(0, n_samples + lag)
                        series_i = state_time_course[n][valid_range, i]
                        series_j = state_time_course[n][-lag:n_samples, j]
                    elif lag == 0:
                        series_i = state_time_course[n][:, i]
                        series_j = state_time_course[n][:, j]
                    else:
                        valid_range = slice(lag, n_samples)
                        series_i = state_time_course[n][valid_range, i]
                        series_j = state_time_course[n][:n_samples - lag, j]
                    
                    # Since the data are binary, we can directly compute mutual information.
                    mi = mutual_info_score(series_i, series_j)

                    if normalize:
                        # Compute entropies of the two series
                        h_i = compute_entropy(series_i, unit=unit)
                        h_j = compute_entropy(series_j, unit=unit)

                        # Compute normalized MI
                        geometric_mean_h = np.sqrt(h_i * h_j)
                        mi = mi / geometric_mean_h if geometric_mean_h > 0 else 0 # avoid division by zero

                    mi_metric[n, idx, i, j] = mi
                    if i != j:
                        mi_metric[n, idx, j, i] = mi

        # Exclude self-information
        if exclude_self:
            total_sum = np.sum(mi_metric, axis=-1)
            diagonal = np.diagonal(mi_metric, axis1=2, axis2=3)
            # shape: (n_subjects, n_lags, n_states)
            avg_mi = (total_sum - diagonal) / (n_states - 1)
        else:
            avg_mi = np.mean(mi_metric, axis=-1)
    
    return avg_mi, mi_metric


def compute_transition_probability(state_time_course):
    """Computes the transition probability matrix from state time courses.

    Parameters
    ----------
    state_time_course : list of np.ndarray or np.ndarray
        State time courses. Shape can be (n_subjects, n_samples, n_states) or
        (n_samples, n_states). If the first dimension is present, it will be
        concatenated across subjects into a single time course.

    Returns
    -------
    transition_probs : np.ndarray
        Transition probability matrix. Shape is (n_states, n_states).
        Each element (i, j) represents the probability of transitioning from
        state i to state j.
    """
    # Validation
    if isinstance(state_time_course, list):
        state_time_course = np.concatenate(state_time_course, axis=0)  # concatenate across subjects
    elif not isinstance(state_time_course, np.ndarray):
        raise ValueError("state_time_course must be a list or a numpy array.")

    # Get data dimensions
    n_states = state_time_course.shape[-1]

    # Get state transitions
    state_transitions = np.argmax(state_time_course, axis=1)

    # Get unique states and index them
    unique_states = np.unique(state_transitions)
    state_to_index = {
        state: idx for idx, state in enumerate(unique_states)
    }

    # Initialize count matrix
    counts = np.zeros((n_states, n_states), dtype=int)

    # Count transitions from each state to the next
    for s1, s2 in zip(state_transitions[:-1], state_transitions[1:]):
        i, j = state_to_index[s1], state_to_index[s2]
        counts[i, j] += 1

    # Convert counts to probabilities
    transition_probs = counts.astype(float)
    row_sums = transition_probs.sum(axis=1, keepdims=True)

    # Avoid division by zero for states with no outgoing transitions
    row_sums[row_sums == 0] = 1
    transition_probs /= row_sums

    return transition_probs
