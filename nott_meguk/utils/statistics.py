"""Functions for statistical testing."""

import warnings
import numpy as np
from pqdm.threads import pqdm
from scipy import stats
from scipy.spatial.distance import cosine

from osl_dynamics.analysis import regression
from osl_dynamics.inference import modes
from utils.analysis import compute_sw_state_time_course
from utils.array_ops import window_shuffle


def _check_stat_assumption(samples1, samples2, ks_alpha=0.05, ev_alpha=0.05):
    """Checks normality of each sample and whether samples have an equal variance.

    Parameters
    ----------
    samples1 : np.ndarray
        Array of sample data (group 1). Shape must be (n_samples,).
    samples2 : np.ndarray
        Array of sample data (group 2). Shape must be (n_samples,).
    ks_alpha : float
        Threshold to use for null hypothesis rejection in the Kolmogorov-Smirnov test.
        Defaults to 0.05.
    ev_alpha : float
        Threshold to use for null hypothesis rejection in the equal variance test.
        This test can be the Levene's test or Bartlett's test, depending on the
        normality of sample distributions. Defaults to 0.05.

    Returns
    -------
    nm_flag : bool
        If True, both samples follow a normal distribution.
    ev_flag : bool
        If True, two sample groups have an equal variance.
    """
    # Set flags for normality and equal variance
    nm_flag, ev_flag = True, True
    print("*** Checking Normality & Equal Variance Assumptions ***")

    # Check normality assumption
    ks_pvals = []
    for s, samples in enumerate([samples1, samples2]):
        stand_samples = stats.zscore(samples)
        res = stats.ks_1samp(stand_samples, cdf=stats.norm.cdf)
        ks_pvals.append(res.pvalue)
        print(f"\t[KS Test] p-value (Sample #{s}): {res.pvalue}")
        if res.pvalue < ks_alpha:
            print(
                f"\t[KS Test] Sample #{s}: Null hypothesis rejected. The data are not distributed "
                + "according to the standard normal distribution."
            )

    # Check equal variance assumption
    if np.sum([pval < ks_alpha for pval in ks_pvals]) != 0:
        nm_flag = False
        # Levene's test
        _, ev_pval = stats.levene(samples1, samples2)
        ev_test_name = "Levene's"
    else:
        # Bartlett's test
        _, ev_pval = stats.bartlett(samples1, samples2)
        ev_test_name = "Bartlett's"
    print(f"\t[{ev_test_name} Test] p-value: ", ev_pval)
    if ev_pval < ev_alpha:
        print(
            f"\t[{ev_test_name} Test] Null hypothesis rejected. The populations do not have equal variances."
        )
        ev_flag = False

    return nm_flag, ev_flag


def stat_ind_two_samples(
    samples1, samples2, alpha=0.05, bonferroni_ntest=None, test=None
):
    """Performs a statistical test comparing two independent samples.

    Parameters
    ----------
    samples1 : np.ndarray
        Array of sample data (group 1). Shape must be (n_samples,).
    samples2 : np.ndarray
        Array of sample data (group 2). Shape must be (n_samples,).
    alpha : float
        Threshold to use for null hypothesis rejection. Defaults to 0.05.
    bonferroni_ntest : int
        Number of tests to be used for Bonferroni correction. Default to None.
    test : str
        Statistical test to use. Defaults to None, which automatically selects
        the test after checking the assumptions.

    Returns
    -------
    stat : float
        The test statistic. The test can be the Student's t-test, Welch's t-test,
        or Wilcoxon Rank Sum test depending on the test assumptions.
    pval : float
        The p-value of the test.
    sig_indicator : bool
        Whether the p-value is significant or not. If bonferroni_ntest is given,
        the p-value will be evaluated against the corrected threshold.
    """
    # Check normality and equal variance assumption
    if test is None:
        nm_flag, ev_flag = _check_stat_assumption(samples1, samples2)
    else:
        if test == "ttest":
            nm_flag, ev_flag = True, True
        elif test == "welch":
            nm_flag, ev_flag = True, False
        elif test == "wilcoxon":
            nm_flag, ev_flag = False, True

    # Compare two independent groups
    print("*** Comparing Two Independent Groups ***")
    if nm_flag and ev_flag:
        print("\tConducting the two-samples independent T-Test ...")
        stat, pval = stats.ttest_ind(samples1, samples2, equal_var=True)
    if nm_flag and not ev_flag:
        print("\tConducting the Welch's t-test ...")
        stat, pval = stats.ttest_ind(samples1, samples2, equal_var=False)
    if not nm_flag:
        print("\tConducting the Wilcoxon Rank Sum test ...")
        if not ev_flag:
            warnings.warn("Caution: Distributions have unequal variances.", UserWarning)
        stat, pval = stats.ranksums(samples1, samples2)
    print(f"\tResult: statistic={stat} | p-value={pval}")

    # Apply Bonferroni correction
    if bonferroni_ntest is not None:
        alpha /= bonferroni_ntest
    sig_indicator = pval < alpha
    print(f"[Bonferroni Correction] Threshold: {alpha}, Significance: {sig_indicator}")

    return stat, pval, sig_indicator


def split_half_permutation_test(
    time_varying_covariances,
    alpha_time_courses,
    cosine_similarities,
    window_length,
    step_size,
    shuffle_window_length,
    n_perms=1000,
    n_jobs=1,
):
    """Performs permutation tests for split-half reproducibility metrics.
    
    Here, we use cosine similarities between power maps for each state as the
    metric of interest.

    Parameters
    ----------
    time_varying_covariances : list of list of np.ndarray
        List containing time-varying covariance matrices for each split.
        Each element should have shape (n_subjects, n_windows, n_channels,
        n_channels).
    alpha_time_courses : list of list of np.ndarray
        List containing alpha time courses for each split.
        Each element should have shape (n_subjects, n_samples, n_states).
    cosine_similarities : np.ndarray
        Array containing cosine similarities between power maps for each state.
        Shape should be (n_states,).
    window_length : int
        Length of the sliding window in samples.
    step_size : int
        Step size for the sliding window in samples.
    shuffle_window_length : int
        Length of the window used to shuffle the data.
    n_perms : int, optional
        Number of permutations to perform. Defaults to 1000.
    n_jobs : int, optional
        Number of parallel jobs to run.
        Defaults to 1 (no parallelization).

    Returns
    -------
    cd_pvals : list
        List of p-values for cosine similarities.
        Each element corresponds to a state.
    cd_sig_indicator : list
        List of significance indicators for cosine similarities.
        Each element corresponds to a state and indicates significance level.
    """
    # Get time-varying covariancesfor each split
    tv_covs_1 = time_varying_covariances[0]
    tv_covs_2 = time_varying_covariances[1]

    # Get number of subjects
    n_subjects_1 = len(tv_covs_1)
    n_subjects_2 = len(tv_covs_2)

    # Get alpha time courses for each split
    alphas_1 = alpha_time_courses[0]
    alphas_2 = alpha_time_courses[1]

    # Get the number of states
    n_states = alphas_1[0].shape[1]

    # Set arguments for sliding window computations
    sw_kwargs = {
        "window_length": window_length,
        "step_size": step_size,
        "shuffle_window_length": shuffle_window_length,
        "n_jobs": 1,
    }

    # Perform split-half permutation test
    def _get_metrics(n):
        # Shuffle the alpha time courses in windows
        rng_1 = np.random.default_rng(n)
        rng_2 = np.random.default_rng(n + 1)

        shuffled_alphas_1 = window_shuffle(
            alphas_1, shuffle_window_length, rng=rng_1
        )
        shuffled_alphas_2 = window_shuffle(
            alphas_2, shuffle_window_length, rng=rng_2
        )

        # Get shuffled state time courses
        shuffled_stc_1 = modes.argmax_time_courses(shuffled_alphas_1)
        shuffled_stc_2 = modes.argmax_time_courses(shuffled_alphas_2)

        # Compute sliding window state time courses
        sw_stcs_1 = compute_sw_state_time_course(shuffled_stc_1, **sw_kwargs)
        sw_stcs_2 = compute_sw_state_time_course(shuffled_stc_2, **sw_kwargs)

        # Regress time-varying covariances on state time courses
        power_1, power_2 = [], []

        for i in range(n_subjects_1):
            pow_1 = regression.linear(
                sw_stcs_1[i],
                np.diagonal(tv_covs_1[i], axis1=1, axis2=2),
                fit_intercept=False,
            )
            power_1.append(pow_1)
        
        for i in range(n_subjects_2):
            pow_2 = regression.linear(
                sw_stcs_2[i],
                np.diagonal(tv_covs_2[i], axis1=1, axis2=2),
                fit_intercept=False,
            )
            power_2.append(pow_2)

        power_1 = np.mean(power_1, axis=0)  # average over subjects
        power_2 = np.mean(power_2, axis=0)  # average over subjects
        # shape: (n_states, n_channels)

        # Demean power maps across states
        power_1 -= np.mean(power_1, axis=0, keepdims=True)
        power_2 -= np.mean(power_2, axis=0, keepdims=True)

        # Compute cosine similarities between power maps
        cs = []
        for p1, p2 in zip(power_1, power_2):
            cs.append(1 - cosine(p1, p2))

        return cs

    perm_args = [n for n in range(n_perms)]

    results = pqdm(
        perm_args,
        _get_metrics,
        n_jobs=n_jobs,
        desc="Split-half Permutations",
    )

    # Build null distributions
    null_cs = np.max(np.stack(results), axis=1)
    # shape: (n_perms, n_states)

    # Get thresholds for statistical significance
    null_cs = null_cs.flatten()

    sig_labels = np.array(["*", "**", "***"])
    cs_thresholds = np.percentile(null_cs, [95, 99, 99.9])

    # Get statistical significance
    cs_pvals = []
    cs_sig_indicator = []
    for i in range(n_states):
        cs_sig = cosine_similarities[i] > cs_thresholds
        cs_pvals.append(np.mean(null_cs > cosine_similarities[i]))

        if np.any(cs_sig):
            cs_sig_indicator.append(sig_labels[cs_sig][-1])
        else:
            cs_sig_indicator.append("n.s.")

    return cs_pvals, cs_sig_indicator
