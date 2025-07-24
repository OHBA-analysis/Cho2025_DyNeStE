"""Functions for pos-hoc analysis."""

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import mutual_info_score
from tqdm import trange
from osl_dynamics.analysis import power, connectivity, tinda
from osl_dynamics.data import Data
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


def compute_sw_state_time_course(
    stc,
    window_length,
    step_size,
    shuffle_window_length=None,
    n_jobs=1,
):
    """Applies sliding window to state time courses.

    Parameters
    ----------
    stc : list of np.ndarray
        List of state time courses for each subject.
        Shape is (n_subjects, n_samples, n_states).
    window_length : int
        Length of the sliding window in samples.
    step_size : int
        Step size for the sliding window in samples.
    shuffle_window_length : int, optional
        Length of the window used to shuffle the data.
    n_jobs : int, optional
        Number of parallel jobs to run.
        Defaults to 1 (no parallelization).

    Returns
    -------
    sw_stc : list of np.ndarray
        List of sliding-window state time courses for each subject.
        Each array has shape (n_windows, n_states).
    """
    # Trim the data if necessary
    if shuffle_window_length is not None:
        for i, tc in enumerate(stc):
            n_samples = tc.shape[0]
            n_windows = n_samples // shuffle_window_length
            stc[i] = tc[:n_windows * shuffle_window_length]

    # Compute sliding window state time courses
    sw_stc = power.sliding_window_power(
        stc,
        window_length=window_length,
        step_size=step_size,
        power_type="mean",
        n_jobs=n_jobs,
    )

    return sw_stc


def compute_tv_covariances(
    data,
    sampling_frequency,
    window_length,
    step_size,
    shuffle_window_length=None,
    n_jobs=1,
):
    """Calculates time-varying covariance matrices.

    Parameters
    ----------
    data : list of np.ndarray
        List of time series data for each subject.
        Shape is (n_subjects, n_samples, n_channels).
    sampling_frequency : int
        Sampling frequency of the data in Hz.
    window_length : int
        Length of the sliding window in samples.
    step_size : int
        Step size for the sliding window in samples.
    shuffle_window_length : int, optional
        Length of the window used to shuffle the data.
    n_jobs : int, optional
        Number of parallel jobs to run.
        Defaults to 1 (no parallelization).

    Returns
    -------
    tv_cov : list of np.ndarray
        List of time-varying covariance matrices for each subject.
        Each matrix has shape (n_windows, n_channels, n_channels).
    """
    # Prepare the data
    dataset = Data(data, sampling_frequency=sampling_frequency)
    dataset.prepare({
        "filter": {"low_freq": 1.5, "high_freq": 40, "use_raw": True},
        "amplitude_envelope": {},
        "moving_average": {"n_window": 25},
        "standardize": {},
    })
    time_series = dataset.time_series()

    # Trim the data if necessary
    if shuffle_window_length is not None:
        for i in range(dataset.n_sessions):
            n_samples = time_series[i].shape[0]
            n_windows = n_samples // shuffle_window_length
            time_series[i] = time_series[i][:n_windows * shuffle_window_length]

    # Compute time-varying covariances
    tv_cov = connectivity.sliding_window_connectivity(
        time_series,
        window_length=window_length,
        step_size=step_size,
        conn_type="cov",
        n_jobs=n_jobs,
    )

    return tv_cov


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


def run_tinda(state_time_courses):
    """Runs the TINDA algorithm on state time courses.

    Parameters
    ----------
    state_time_courses : list of np.ndarray
        State time courses for each subject.
        Shape is (n_subjects, n_samples, n_states).

    Returns
    -------
    fo_density : np.ndarray
        Fractional occupancies of the states in each interval bins
        across subjects. Shape is (n_interval_states, n_density_states,
        n_bins, n_interval_ranges, n_subjects).
    tinda_stats : tuple of list of dict
        Statistics from TINDA.
        Shape is (n_subjects, n_states, n_features).
    best_sequence : np.ndarray
        Array of the best circular sequence of states.
    asymmetry_matrix : np.ndarray
        Asymmetry matrix computed from the fractional occupancies.
        Shape is (n_states, n_states, n_subjects).
    """
    # Run TINDA
    fo_density, _, tinda_stats = tinda.tinda(state_time_courses)
    # shape (fo_density): (n_interval_states, n_density_states, n_bins, 
    #                      n_interval_ranges, n_subjects)

    # Find the best circular sequence
    best_sequence = tinda.optimise_sequence(fo_density)
    # shape: (n_states,)

    # Compute the asymmetry matrix
    asymmetry_matrix = np.squeeze(
        np.nanmean((fo_density[:, :, 0] - fo_density[:, :, 1]), axis=2)
    )
    asymmetry_matrix[np.isnan(asymmetry_matrix)] = 0
    # shape: (n_states, n_states, n_subjects)
    
    return fo_density, tinda_stats, best_sequence, asymmetry_matrix


def run_tinda_quintile(state_time_courses, interval_mode, interval_range):
    """Runs the TINDA algorithm on state time courses.

    Parameters
    ----------
    state_time_courses : list of np.ndarray
        State time courses for each subject.
        Shape is (n_subjects, n_samples, n_states).

    Returns
    -------
    fo_density : np.ndarray
        Fractional occupancies of the states in each interval bins
        across subjects. Shape is (n_interval_states, n_density_states,
        n_bins, n_interval_ranges, n_subjects).
    tinda_stats : tuple of list of dict
        Statistics from TINDA.
        Shape is (n_subjects, n_states, n_features).
    best_sequences : list of np.ndarray
        List of the best circular sequences of states for each interval.
    asymmetry_matrices : list of np.ndarray
        List of asymmetry matrices for each interval, computed from the
        fractional occupancies.
        Shape is (n_interval_ranges, n_states, n_states, n_subjects).
    """
    # Validate inputs
    n_qunitiles = 5  # by the definition of quintiles
    if len(interval_range) - 1 != n_qunitiles:
        raise ValueError("Interval range must have 6 values for quintile analysis.")

    # Run TINDA
    fo_density, _, tinda_stats = tinda.tinda(
        state_time_courses,
        interval_mode=interval_mode,
        interval_range=interval_range,
    )
    # shape (fo_density): (n_interval_states, n_density_states, n_bins, 
    #                      n_interval_ranges, n_subjects)

    # Find the best circular sequences
    best_sequences = []
    for i in range(n_qunitiles):
        best_sequences.append(tinda.optimise_sequence(fo_density[:, :, :, i:i+1]))
    # shape: (n_interval_ranges, n_states)

    # Compute the interval-wise asymmetry matrix
    asymmetry_matrices = []
    for i in range(n_qunitiles):
        asymmetry_matrix = np.squeeze(
            np.nanmean((fo_density[:, :, 0, i:i+1] - fo_density[:, :, 1, i:i+1]), axis=2)
        )
        asymmetry_matrix[np.isnan(asymmetry_matrix)] = 0
        # shape: (n_states, n_states, n_subjects)
        asymmetry_matrices.append(asymmetry_matrix)
    # shape: (n_interval_ranges, n_states, n_states, n_subjects)
    
    return fo_density, tinda_stats, best_sequences, asymmetry_matrices


def find_strongest_edges(
    fo_density,
    asymmetry_matrix,
    state_sequence=None,
    method="statistics",
):
    """Finds the strongest edges in the asymmetry matrix based on
       the specified method.

    Parameters
    ----------
    fo_density : np.ndarray
        Fractional occupancy density matrix.
        Shape is (n_interval_states, n_density_states, n_bins,
        n_interval_ranges, n_subjects).
    asymmetry_matrix : np.ndarray
        Asymmetry matrix computed from the fractional occupancies.
        Shape is (n_states, n_states, n_subjects).
    state_sequence : np.ndarray, optional
        Sequence of states to consider for finding edges.
        If provided, the asymmetry matrix will be reordered
        according to this sequence.
    method : str, optional
        Method to use for finding edges. Can be either
        "statistics" (default) or "percentile".

    Returns
    -------
    edges : np.ndarray
        Binary matrix indicating the presence of edges.
        Shape is (n_states, n_states).
    edge_idx : np.ndarray
        Indices of the edges found. Shape is (n_edges, 2).
    """
    # Preprocess state sequence
    if state_sequence is not None:
        state_sequence = np.concatenate((
            [state_sequence[0]], state_sequence[1:][::-1]
        ))  # change from counter-clockwise to clockwise

    # Find the strongest edges based on the specified method
    if method == "statistics":
        # Validate inputs
        if fo_density.ndim != 5:
            raise ValueError("fo_density matrix must have 5 dimensions.")

        # Perform t-test on interval asymmetries
        n = np.prod(asymmetry_matrix.shape[:2]) - asymmetry_matrix.shape[0]
        thr = 0.05 / n  # Bonferroni-corrected threshold
        print(f"Bonferroni correction (n={n}) with threshold: {thr:.4f}")

        group1 = np.squeeze(fo_density[:, :, 0])
        group2 = np.squeeze(fo_density[:, :, 1])
        # shape: (n_states, n_states, n_subjects)

        if state_sequence is not None:
            group1 = group1[np.ix_(state_sequence, state_sequence, np.arange(group1.shape[2]))]
            group2 = group2[np.ix_(state_sequence, state_sequence, np.arange(group2.shape[2]))]

        n_states, _, _ = group1.shape
        edges = np.zeros((n_states, n_states))

        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    _, p_val = ttest_rel(group1[i, j, :], group2[i, j, :], nan_policy="omit")
                    if p_val < thr:
                        edges[i, j] = 1

        edge_idx = np.argwhere(edges == 1)

    if method == "percentile":
        # Validate inputs
        if asymmetry_matrix.ndim == 3:
            asymmetry_matrix = np.nanmean(asymmetry_matrix, axis=2)  # average across subjects
        elif asymmetry_matrix.ndim != 2:
            raise ValueError("asymmetry_matrix must have 2 or 3 dimensions.")
        
        # Reorder matrix by state sequence
        if state_sequence is not None:
            asymmetry_matrix = asymmetry_matrix[np.ix_(state_sequence, state_sequence)]

        # Find indices of largest 25% values (excluding diagonal)
        percentile = 0.25
        n_edges = int((asymmetry_matrix.shape - asymmetry_matrix.shape[0]) * percentile)
        edge_idx = np.abs(asymmetry_matrix).argsort(axis=None)[-n_edges:]

        # Get edges by setting the largest indices to 1
        edges = np.zeros(asymmetry_matrix.shape)
        edges[np.unravel_index(edge_idx, asymmetry_matrix.shape)] = 1
        edge_idx = np.stack(np.unravel_index(edge_idx, asymmetry_matrix.shape), axis=-1)
        # convert to 2D indices

    return edges, edge_idx


def get_cycle_strengths(asymmetry_matrix, state_sequence):
    """Calculates the cycle strengths for a given state sequence.

    Parameters
    ----------
    asymmetry_matrix : np.ndarray
        Asymmetry matrix computed from the fractional occupancies.
        Shape is (n_states, n_states, n_subjects).
    state_sequence : np.ndarray
        Cyclic sequence of the states. Shape is (n_states,).

    Returns
    -------
    cycle_strengths : np.ndarray
        Cycle strengths for each subject. Shape is (n_subjects,).
    """
    # Compute cycle strengths
    angleplot = tinda.circle_angles(state_sequence)  # shape: (n_states, n_states)
    cycle_strengths = tinda.compute_cycle_strength(
        angleplot, asymmetry_matrix
    )  # shape: (n_subjects,)
    return cycle_strengths


def compute_interval_durations(
    tinda_stats,
    interval_range,
    sampling_frequency
):
    """Computes the subject-level interval durations for each quintile.

    Parameters
    ----------
    tinda_stats : tuple of list of dict
        Statistics from TINDA.
        Shape is (n_subjects, n_states, n_features).
    interval_range : list or np.ndarray
        List of interval ranges for quintiles.
        Should have 6 values for quintile analysis.
    sampling_frequency : int
        Sampling frequency of the data in Hz.

    Returns
    -------
    mean_interval_durations : np.ndarray
        Mean interval durations for each subject and quintile.
        Shape is (n_subjects, n_quintiles).
    """
    # Get the data dimensions
    n_subjects = len(tinda_stats)
    n_states = len(tinda_stats[0])
    n_quintiles = len(interval_range) - 1  # number of quintiles
    
    # Compute subject-level interval durations
    interval_durations = []
    for n in range(n_subjects):
        durations_state_quintile = []  # interval durations for each state and quintile
        for s in range(n_states):
            # Get instances of interval durations (in seconds)
            durations = tinda_stats[n][s]["durations"] / sampling_frequency
            
            # Compute percentiles
            perc = np.unique(np.percentile(durations, interval_range))  # np.unique() sorts automatically

            # Bin the durations based on percentiles
            binned_indices = np.digitize(durations, perc, right=False)
            # NOTE: The indices will range from 1 to len(percentiles)-1 corresponding to
            #       the bins (0-20%), (20-40%), (40-60%), (60-80%), (80-100%).
            binned_indices = np.clip(binned_indices, 1, len(perc) - 1)
            # NOTE: This handles the edge case where minimum and maximum values might get off-indices.
            #       This ensures that values equal to the min are in Bin #1, and values equal to the max
            #       are in the last bin.
            binned_indices -= 1  # for 0-based indexing

            # Create a list of durations for each quintile
            durations_quintile = []
            for i in range(len(perc) - 1):
                mask = (binned_indices == i)
                durations_quintile.append(durations[mask])
            # shape: (n_quintiles, n_durations)
            
            durations_state_quintile.append(durations_quintile)
            # shape: (n_states, n_quintiles, n_durations)

        interval_durations.append(durations_state_quintile)
        # shape: (n_subjects, n_states, n_quintiles, n_durations)

    # Average interval durations over duration instances and states
    mean_interval_durations = np.zeros((n_subjects, n_states, n_quintiles))
    for n in range(n_subjects):
        for s in range(n_states):
            for q in range(n_quintiles):
                mean_interval_durations[n, s, q] = np.mean(
                    interval_durations[n][s][q]
                )  # average across duration instances
                # shape: (n_subjects, n_states, n_quintiles)
    mean_interval_durations = np.mean(mean_interval_durations, axis=1)  # average across states
    # shape: (n_subjects, n_quintiles)

    return mean_interval_durations
