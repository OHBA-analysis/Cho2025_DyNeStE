"""Functions for data visualization and plotting.

This module contains several functions from osl_dynamics.utils.plotting.py,
which has been edited for the purpose of this project.
"""

import os
import mne
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import MaxNLocator, ScalarFormatter
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils.plotting import plot_line
from osl_dynamics.utils.misc import override_dict_defaults
from osl_dynamics.utils.plotting import create_figure
from utils.array_ops import round_nonzero_decimal, round_up_half


def save(fig, filename, **kwargs):
    """Saves a matplotlib figure to a file.

    Parameters
    ----------
    fig : plt.figure
        Matplotlib figure object.
    filename : str
        Output filename.
    kwargs : dict, optional
        Additional arguments to pass to `fig.savefig <https://matplotlib.org\
        /stable/api/_as_gen/matplotlib.figure.Figure.savefig.html>`_.
    """
    if not filename.endswith(".png"):
        filename += ".png"
    fig.savefig(filename, dpi=300, bbox_inches="tight", **kwargs)
    plt.close(fig)


def _format_colorbar_ticks(ax):
    """Formats x-axis ticks in the colobar such that integer values are 
       plotted, instead of decimal values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A colobar axis to format.
    """

    if np.any(np.abs(ax.get_xlim()) < 1):
        hmin = round_nonzero_decimal(ax.get_xlim()[0], method="ceil") # ceiling for negative values
        hmax = round_nonzero_decimal(ax.get_xlim()[1], method="floor") # floor for positive values
        ax.set_xticks(np.array([hmin, 0, hmax]))
    else:
        ax.set_xticks(
            [round_up_half(val) for val in ax.get_xticks()[1:-1]]
        )
    
    return None


def _get_color_tools(name, n_colors):
    """Creates a color palette and corresponding line styles nad markers
       for plotting.

    Parameters
    ----------
    name : string
        Name of the colormap to use.
    n_colors : int
        Number of colors to use.

    Returns
    -------
    palette : list
        List of colors.
    linestyles : list
        List of linestyles corresponding to each color.
    markers : list
        List of markers corresponding to each color.
    """
    # Validation
    if n_colors not in [6, 12]:
        raise ValueError("Number of colors must be either 6 or 12.")
    
    # Get color palette, line styles, and markers
    if name == "tol_bright":
        tol_bright = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]
        if n_colors == 6:
            palette = tol_bright.copy()
            linestyles = ["solid"] * n_colors
            markers = ["o"] * n_colors
        elif n_colors == 12:
            palette = np.tile(tol_bright.copy(), 2)
            linestyles = ["solid"] * 6 + ["dashed"] * 6
            markers = ["o"] * 6 + ["d"] * 6

    return palette, linestyles, markers


def _categorize_pvalue(p_val, bonferroni_n_tests=1):
    """Assigns a label indicating statistical significance that corresponds 
    to an input p-value.

    Parameters
    ----------
    p_val : float
        P-value from a statistical test.
    bonferroni_n_tests : int, optional
        Number of tests performed for Bonferroni correction. Defaults to 1.

    Returns
    -------
    p_label : str
        Label representing a statistical significance.
    """ 

    # Define thresholds and labels
    thresholds = np.array([1e-3, 0.01, 0.05]) / bonferroni_n_tests
    labels = ["***", "**", "*", "n.s."]

    # Check if p-value is within the thresholds
    arr_to_compare = np.concat((thresholds, [p_val]))
    ordinal_idx = np.max(np.where(np.sort(arr_to_compare) == p_val)[0])
    # NOTE: use maximum for the case in which a p-value and threshold are identical
    p_label = labels[ordinal_idx]

    return p_label


def plot_alpha(
    *alpha,
    n_samples=None,
    cmap="tab10",
    sampling_frequency=None,
    y_labels=None,
    title=None,
    fontsize=15,
    plot_kwargs=None,
    fig_kwargs=None,
    filename=None,
    axes=None,
):
    """Plot alpha.

    Parameters
    ----------
    alpha : np.ndarray
        A collection of alphas passed as separate arguments.
    n_samples: int, optional
        Number of time points to be plotted.
    cmap : str or matplotlib.colors.ListedColormap, optional
        Matplotlib colormap.
    sampling_frequency : float, optional
        The sampling frequency of the data in Hz.
    y_labels : str, optional
        Labels for the y-axis of each alpha time series.
    title : str, optional
        Title for the plot.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 15.
    plot_kwargs : dict, optional
        Any parameters to be passed to `plt.stackplot <https://matplotlib.org\
        /stable/api/_as_gen/matplotlib.pyplot.stackplot.html>`_.
    fig_kwargs : dict, optional
        Arguments to pass to :code:`plt.subplots()`.
    filename : str, optional
        Output filename.
    axes : list of plt.axes, optional
        A list of matplotlib axes to plot on. If :code:`None`, a new
        figure is created.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`ax=None` and
        :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """
    n_alphas = len(alpha)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    if axes is not None and len(axes) != n_alphas:
        raise ValueError("Number of axes must match number of alphas.")

    n_modes = max(a.shape[1] for a in alpha)
    n_samples = min(n_samples or np.inf, alpha[0].shape[0])
    if isinstance(cmap, str):
        if cmap in [
            "Pastel1",
            "Pastel2",
            "Paired",
            "Accent",
            "Dark2",
            "Set1",
            "Set2",
            "Set3",
            "tab10",
            "tab20",
            "tab20b",
            "tab20c",
        ]:
            cmap = plt.cm.get_cmap(name=cmap)
        else:
            cmap = plt.cm.get_cmap(name=cmap, lut=n_modes)
    cmap = cmap.copy()
    colors = cmap.colors

    # Validation
    if fig_kwargs is None:
        fig_kwargs = {}
    default_fig_kwargs = dict(
        figsize=(12, 2.5 * n_alphas), sharex="all", facecolor="white"
    )
    fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    default_plot_kwargs = dict(colors=colors)
    plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    if y_labels is None:
        y_labels = [None] * n_alphas
    elif isinstance(y_labels, str):
        y_labels = [y_labels] * n_alphas
    elif len(y_labels) != n_alphas:
        raise ValueError("Incorrect number of y_labels passed.")

    # Create figure if axes not passed
    if axes is None:
        fig, axes = create_figure(n_alphas, **fig_kwargs)
    else:
        fig = axes[0].get_figure()

    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Plot data
    for a, ax, y_label in zip(alpha, axes, y_labels):
        time_vector = (
            np.arange(n_samples) / sampling_frequency
            if sampling_frequency
            else range(n_samples)
        )
        ax.stackplot(time_vector, a[:n_samples].T, **plot_kwargs)
        ax.autoscale(tight=True)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

    # Set axis label and title
    axes[-1].set_xlabel("Time (s)" if sampling_frequency else "Sample", fontsize=fontsize)
    axes[0].set_title(title)

    # Fix layout
    plt.tight_layout()

    # Add a colour bar
    norm = matplotlib.colors.BoundaryNorm(
        boundaries=range(n_modes + 1), ncolors=n_modes
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(right=0.94)
    cb_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    cb = fig.colorbar(mappable, cax=cb_ax, ticks=np.arange(0.5, n_modes, 1))
    cb.ax.set_yticklabels(range(1, n_modes + 1))

    # Save to file if a filename has been passed
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, axes


def plot_replay_evoked_state_activations(
    data,
    fontsize=14,
    filename=None,
):
    """Plots replay-evoked state activations.
    
    Parameters
    ----------
    data : np.ndarray
        Data containing replay-evoked state activations. Shape must be
        (n_sessions, n_epoched_samples, n_states).
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 14.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`ax=None` and
        :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """
    # Validation
    if data.ndim != 3:
        raise ValueError(
            "Data should be a 3-D array of shape (n_sessions, n_epoched_samples, n_states)."
        )

    # Get data moments
    data_avg = np.mean(data, axis=0)  # average across sessions
    data_ste = np.std(data, axis=0) / np.sqrt(len(data))  # standard error of the mean
    # shape: (n_epoched_samples, n_states)

    # Get data dimensions
    n_samples, n_states = data_avg.shape

    # Get data ranges
    vmax = (data_avg + data_ste).max()
    vmin = (data_avg - data_ste).min()
    vrange = vmax - vmin
    vline = vmin - 0.01 * vrange  # y-value of the line to indicate significant clusters

    # Set visualization hyperparameters
    palette, color_ls, _ = _get_color_tools("tol_bright", n_states)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8))
    x_time = np.arange(n_samples)
    for s in range(n_states):
        ax.plot(
            x_time,
            data_avg[:, s],
            color=palette[s],
            linewidth=2,
            linestyle=color_ls[s],
            label=f"State {s + 1}",
        )
        ax.fill_between(
            x_time, data_avg[:, s] - data_ste[:, s], data_avg[:, s] + data_ste[:, s],
            color=palette[s], alpha=0.2,
        )
    ax.axvline(n_samples // 2, color="black", linestyle="dotted", linewidth=2)

    # Perform cluster-based permutation test
    thr = 0.05 / n_states  # Bonferroni-corrected threshold
    for s in range(n_states):
        _, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(
            data[:, :, s],
            threshold=3,
            n_permutations=5000,
            tail=1,
        )
        if any(cluster_pv < thr):
            vline -= 2.5e-2 * vrange
        if len(clusters) > 0:
            for c, cluster in enumerate(clusters):
                p_values = cluster_pv[c]
                if p_values < thr:
                    ax.hlines(
                        y=vline, xmin=cluster[0][0], xmax=cluster[0][-1],
                        linewidth=2, color=palette[s],
                    )
                    ax.text(
                        0 - (n_samples * 0.03), vline, f"{s + 1}", color="k",
                        ha="right", va="center", fontsize=(fontsize - 2),
                    )

    # Adjust axis labels
    ax.set_xlabel("Time (s) from Replay Event", fontsize=fontsize)
    ax.set_ylabel("Change in RSN-State Probabilities", fontsize=fontsize)
    ax.set_title("Replay-Evoked Network Activations", fontsize=(fontsize + 1))

    # Adjust tick settings
    ax.spines[["left", "bottom", "right", "top"]].set_linewidth(1.5)
    ax.set_xticks([0, n_samples // 2, n_samples])
    ax.set_xticklabels([-0.5, 0, 0.5])
    ax.set_xlim([0 - (n_samples * 0.09), n_samples + (n_samples * 0.05)])
    ax.set_ylim([vmin - 0.13 * vrange, 0.15])
    ax.tick_params(axis="both", which="both", width=1.5, labelsize=fontsize)
    
    fig.subplots_adjust(
        left=0.1815, bottom=0.0853, right=0.9750, top=0.9508, wspace=0.2, hspace=0.2
    )

    # Save to file if a filename has been passed
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, ax
    

def plot_network_dependent_replay_intervals(
    data,
    sampling_frequency,
    ylims=None,
    fontsize=14,
    n_jobs=1,
    filename=None,
):
    """Plots network-dependent replay intervals.

    Parameters
    ----------
    data : dict
        Dictionary containing replay intervals for each RSN state.
        Keys are state names and values are lists of replay intervals
        (averaged over replay instances) in samples. It should include
        an additional state "Mean" that contains the overall mean replay
        interval.
    sampling_frequency : int
        Sampling frequency of the data in Hz.
    ylims : list, optional
        List of two floats specifying the y-axis limits.
        Defaults to None.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 14.
    n_jobs : int, optional
        Number of jobs to use for parallel processing during the statistical
        test. Defaults to 1.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`ax=None` and
        :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """ 
    # Create a DataFrame
    flattened_data = [
        (key, value / sampling_frequency)  # convert to seconds
        for key, values in data.items()
        for value in values if value
    ]
    df = pd.DataFrame(flattened_data, columns=["States", "Intervals"])
    n_states = int(df["States"].nunique()) - 1

    # Set colormap
    palette, _, _ = _get_color_tools("tol_bright", n_states)

    # Visualize network-dependent replay intervals
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    overall_mean = df[df["States"] == "Mean"]["Intervals"].mean()

    sns.stripplot(
        data=df, x="States", y="Intervals",
        hue="States", palette=list(palette) + ["tab:orange"],
        marker="o", size=5, legend=False, alpha=0.6,
        zorder=1, ax=ax,
    )
    sns.boxplot(
        data=df, x="States", y="Intervals",
        color="k", width=0.6, gap=0.1,
        showmeans=True, meanprops={
            "marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"
        },
        fill=False, legend=False, zorder=2, ax=ax,
    )
    ax.axhline(
        overall_mean, color="k", linestyle="--", linewidth=1.5, alpha=0.7
    )

    # Perform one-sample max-t permutation test
    sig_idx = np.zeros((n_states,), dtype=int)
    p_vals = np.ones((n_states,))

    thr = 0.05 / n_states  # Bonferroni-corrected threshold
    for s in range(n_states):
        test_data = df[df["States"] == (s + 1)]["Intervals"].values
        rest_mean = df[~df["States"].isin([s + 1, "Mean"])]["Intervals"].mean()
        test_data -= rest_mean  # demean data
        _, p_val, _ = mne.stats.permutation_t_test(
            test_data[:, np.newaxis],
            n_permutations=10000,
            tail=-1,  # mean less than zero
            n_jobs=n_jobs,
        )
        if p_val < thr:
            sig_idx[s], p_vals[s] = 1, p_val
            print(f"Significant state: {s + 1} | p-value: {p_val}")

    # Add significance labels
    for s in range(n_states):
        if sig_idx[s]:
            p_label = _categorize_pvalue(p_vals[s], bonferroni_n_tests=n_states)
            ax.text(
                s, ax.get_ylim()[1] * 0.97, p_label,
                color="k", ha="center", va="bottom",
                fontweight="bold", fontsize=fontsize,
            )

    # Adjust axis settings
    if ylims is not None:
        ax.set_ylim(ylims)
    else:
        vmin, vmax = ax.get_ylim()
        gap = (vmax - vmin) * 0.05
        ax.set_ylim([vmin, vmax + gap])
    ax.set_xlabel("RSN States", fontsize=fontsize)
    ax.set_ylabel("Replay Intervals (s)", fontsize=fontsize)
    ax.set_title("Replay Intervals Given Active RSN States", fontsize=(fontsize + 1))
    ax.tick_params(axis="both", which="both", width=1.5, labelsize=fontsize)
    ax.spines[["left", "bottom", "top", "right"]].set_linewidth(1.5)
    
    # Save to file if a filename has been passed
    plt.tight_layout()
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, ax
    

def plot_network_dependent_replay_rates(
    data,
    ylims=None,
    fontsize=14,
    n_jobs=1,
    filename=None,
):
    """Plots network-dependent replay rates.

    Parameters
    ----------
    data : dict
        Dictionary containing replay rates for each RSN state.
        Keys are state names and values are lists of replay rates in
        samples. It should include an additional state "Mean" that
        contains the overall mean replay rates.
    ylims : list, optional
        List of two floats specifying the y-axis limits.
        Defaults to None.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 14.
    n_jobs : int, optional
        Number of jobs to use for parallel processing during the statistical
        test. Defaults to 1.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`ax=None` and
        :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """ 
    # Create a DataFrame
    flattened_data = [
        (key, value)
        for key, values in data.items()
        for value in values if value
    ]
    df = pd.DataFrame(flattened_data, columns=["States", "Replay Rates"])
    n_states = int(df["States"].nunique()) - 1

    # Set colormap
    palette, _, _ = _get_color_tools("tol_bright", n_states)

    # Visualize network-dependent replay rates
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    overall_mean = df[df["States"] == "Mean"]["Replay Rates"].mean()

    sns.stripplot(
        data=df, x="States", y="Replay Rates",
        hue="States", palette=list(palette) + ["tab:orange"],
        marker="o", size=5, legend=False, alpha=0.6,
        zorder=1, ax=ax,
    )
    sns.boxplot(
        data=df, x="States", y="Replay Rates",
        color="k", width=0.6, gap=0.1,
        showmeans=True, meanprops={
            "marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"
        },
        fill=False, legend=False, zorder=2, ax=ax,
    )
    ax.axhline(
        overall_mean, color="k", linestyle="--", linewidth=1.5, alpha=0.7
    )

    # Perform one-sample max-t permutation test
    sig_idx = np.zeros((n_states,), dtype=int)
    p_vals = np.ones((n_states,))

    thr = 0.05 / n_states  # Bonferroni-corrected threshold
    for s in range(n_states):
        test_data = df[df["States"] == (s + 1)]["Replay Rates"].values
        rest_mean = df[~df["States"].isin([s + 1, "Mean"])]["Replay Rates"].mean()
        test_data -= rest_mean  # demean data
        _, p_val, _ = mne.stats.permutation_t_test(
            test_data[:, np.newaxis],
            n_permutations=10000,
            tail=1,  # mean above zero
            n_jobs=n_jobs,
        )
        if p_val < thr:
            sig_idx[s], p_vals[s] = 1, p_val
            print(f"Significant state: {s + 1} | p-value: {p_val}")

    # Add significance labels
    for s in range(n_states):
        if sig_idx[s]:
            p_label = _categorize_pvalue(p_vals[s], bonferroni_n_tests=n_states)
            ax.text(
                s, ax.get_ylim()[1] * 0.97, p_label,
                color="k", ha="center", va="bottom",
                fontweight="bold", fontsize=fontsize,
            )

    # Adjust axis settings
    if ylims is not None:
        ax.set_ylim(ylims)
    else:
        vmin, vmax = ax.get_ylim()
        gap = (vmax - vmin) * 0.05
        ax.set_ylim([vmin, vmax + gap])
    ax.set_xlabel("RSN States", fontsize=fontsize)
    ax.set_ylabel("Replay Rates (/s)", fontsize=fontsize)
    ax.set_title("Replay Rates Given Active RSN States", fontsize=(fontsize + 1))
    ax.tick_params(axis="both", which="both", width=1.5, labelsize=fontsize)
    ax.spines[["left", "bottom", "top", "right"]].set_linewidth(1.5)
    
    # Save to file if a filename has been passed
    plt.tight_layout()
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, ax


def plot_fano_factors(
    fano_factors,
    window_lengths,
    sampling_frequency,
    sig_indices=None,
    ylims=None,
    fontsize=14,
    filename=None,
):
    """Plots Fano factors for the state time courses.

    Parameters
    ----------
    fano_factors : np.ndarray
        Fano factors for the state time courses. Shape must be either
        (n_windows, n_states) or (n_sessions, n_windows, n_states).
    window_lengths : np.ndarray
        Array of window lengths used for Fano factor calculation.
        Shape must be (n_windows,).
    sampling_frequency : int
        Sampling frequency.
    sig_indices : list or np.ndarray, optional
        Indices for significant Fano factors. If provided, the
        corresponding window lengths will be highlighted. Defaults to None.
    ylims : list, optional
        List of two floats specifying the y-axis limits.
        Defaults to None.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 14.
    filename : str, optional
        Output filename. Defaults to None.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object. Only returned if :code:`filename=None`.
    """
    # Validation
    if fano_factors.ndim == 3:
        fano_factors = np.mean(fano_factors, axis=0)  # average over sessions
        # shape: (n_windows, n_states)
    if fano_factors.ndim != 2:
        raise ValueError("Fano factors should be either a 2-D or 3-D array.")
    if fano_factors.shape[0] != len(window_lengths):
        raise ValueError(
            "Fano factors should have the same number of windows as window_lengths."
        )

    if ylims is None:
        min_fano, max_fano = np.min(fano_factors), np.max(fano_factors)
        gap = (max_fano - min_fano) * 0.1
        ylims = [min_fano - gap, max_fano + gap]

    # Get the number of states
    n_states = fano_factors.shape[1]

    # Set colormap
    palette, color_ls, _ = _get_color_tools("tol_bright", n_states)
    
    # Visualize Fano factors
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.6, 8))
    x_windows = (window_lengths / sampling_frequency) * 1000 # samples to milliseconds
    for i in range(n_states):
        ax.plot(
            x_windows, fano_factors[:, i],
            linestyle=color_ls[i], color=palette[i],
        )
        
    if sig_indices is not None:
        y_lower = np.ones(len(x_windows)) * ylims[0]
        y_upper = np.ones(len(x_windows)) * ylims[1]
        ax.fill_between(
            x_windows[sig_indices],
            y_lower[sig_indices],
            y_upper[sig_indices],
            color="tab:red",
            edgecolor="none",
            alpha=0.1,
        )
    
    # Adjust axis settings
    ax.set_xscale("log")
    ax.set_xlabel("Window Length (ms)", fontsize=fontsize)
    ax.set_ylabel("Fano Factor", fontsize=fontsize)
    ax.set_title("Fano Factor of RSN States", fontsize=(fontsize + 1))
    ax.set_ylim(ylims)
    ax.tick_params(axis="both", which="both", width=1.5, labelsize=fontsize)
    ax.spines[["left", "bottom", "right", "top"]].set_linewidth(1.5)
    
    # Save to file if a filename has been passed
    plt.tight_layout()
    if filename is not None:
        save(fig, filename)
    else:
        return fig, ax


def plot_fano_factor_effect_size(
    fano_factors,
    window_lengths,
    sampling_frequency,
    sig_indices=None,
    ylims=None,
    fontsize=14,
    filename=None,
):
    """Plots the effect size (i.e., mean group difference) of Fano factors.

    Here, we compute the difference between two models, and subjects make
    each group.

    Parameters
    ----------
    fano_factors : list of np.ndarray
        Fano factors for the state time courses. Shape of each array
        must be (n_sessions, n_windows, n_states).
    window_lengths : np.ndarray
        Array of window lengths used for Fano factor calculation.
        Shape must be (n_windows,).
    sampling_frequency : int
        Sampling frequency.
    sig_indices : list or np.ndarray, optional
        Indices for significant Fano factors. If provided, the
        corresponding window lengths will be highlighted. Defaults to None.
    ylims : list, optional
        List of two floats specifying the y-axis limits.
        Defaults to None.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 14.
    filename : str, optional
        Output filename. Defaults to None.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object. Only returned if :code:`filename=None`.
    """
    # Validation
    if len(fano_factors) != 2:
        raise ValueError("Two Fano factors should be passed for comparison.")

    for fano in fano_factors:
        if fano.ndim != 3:
            raise ValueError("Fano factors should be a 3-D array.")
        if fano.shape[1] != len(window_lengths):
            raise ValueError(
                "Fano factors should have the same number of windows as window_lengths."
            )

    # Unpack Fano factors        
    fano_1 = fano_factors[0]
    fano_2 = fano_factors[1]
    n1 = fano_1.shape[0]
    n2 = fano_2.shape[0]

    # Get the number of states
    n_states = fano_1.shape[2]

    # Get the group mean Fano factors over sessions
    group_mean_fano1 = np.mean(fano_1, axis=0)  # shape: (n_windows, n_states)
    group_mean_fano2 = np.mean(fano_2, axis=0)  # shape: (n_windows, n_states)
    group_mean_diff = group_mean_fano1 - group_mean_fano2

    # Get the varaince over sessions
    var_fano1 = np.var(fano_1, axis=0, ddof=1)  # shape: (n_windows, n_states)
    var_fano2 = np.var(fano_2, axis=0, ddof=1)  # shape: (n_windows, n_states)
    
    # Calculate the standard error of the mean group difference
    sem_diff = np.sqrt(var_fano1 / n1 + var_fano2 / n2)
    # shape: (n_windows, n_states)

    if ylims is None:
        min_fano = np.min(group_mean_diff - sem_diff),
        max_fano = np.max(group_mean_diff + sem_diff)
        gap = (max_fano - min_fano) * 0.1
        ylims = [min_fano - gap, max_fano + gap]

    # Set colormap
    palette, color_ls, _ = _get_color_tools("tol_bright", n_states)
    
    # Visualize Fano factors
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 6))
    ax.set_xscale("log")  # use log scale

    # Get x-axis values
    x_windows = (window_lengths / sampling_frequency) * 1000  # samples to milliseconds
    n_windows = len(x_windows)
    
    # Jitter x-axis values to avoid overlap
    jitter_strength = 0.11
    jitter_offset = 0.05
    jitter = np.linspace(
        -jitter_strength, jitter_strength, n_states
    ).reshape(1, -1)
    x_jittered = 10 ** (
        np.log10(x_windows).reshape(-1, 1) + jitter + jitter_offset
    )  # shape: (n_windows, n_states)

    # Downsample for better visibility if necessary
    if n_windows > 20:
        tick_indices = np.arange(0, n_windows, 5)[1:]
        x_jittered = x_jittered[tick_indices, :]
        group_mean_diff = group_mean_diff[tick_indices, :]
        sem_diff = sem_diff[tick_indices, :]

    for i in range(n_states):
        ax.errorbar(
            x_jittered[:, i],
            group_mean_diff[:, i],
            yerr=None,
            fmt="o",
            color=palette[i],
            linestyle="none",
            capsize=0,
        )
        # Manually draw error bars with linestyles
        for x, y, err in zip(
            x_jittered[:, i],
            group_mean_diff[:, i],
            sem_diff[:, i],
        ):
            ax.plot(
                [x, x], [y - err, y + err],
                color=palette[i], linestyle=color_ls[i],
                linewidth=1.5,
            )

    if sig_indices is not None:
        y_lower = np.ones(len(x_windows)) * ylims[0]
        y_upper = np.ones(len(x_windows)) * ylims[1]
        ax.fill_between(
            x_windows[sig_indices],
            y_lower[sig_indices],
            y_upper[sig_indices],
            color="tab:red",
            edgecolor="none",
            alpha=0.1,
        )
    
    # Adjust axis settings
    ax.set_xlabel("Window Length (ms)", fontsize=fontsize)
    ax.set_ylabel("Mean Group Difference (a.u.)", fontsize=fontsize)
    ax.set_title("Effect Size of Difference in Fano Factors", fontsize=(fontsize + 1))
    ax.set_ylim(ylims)
    ax.tick_params(axis="both", which="major", width=1.5, labelsize=fontsize)
    ax.minorticks_off()
    ax.spines[["left", "bottom", "right", "top"]].set_linewidth(1.5)

    # Adjust tick settings
    ax.set_xticks(x_windows[tick_indices])
    ax.set_xticklabels([f"{x:.0f}" for x in x_windows[tick_indices]])
    
    # Save to file if a filename has been passed
    plt.tight_layout()
    if filename is not None:
        save(fig, filename)
    else:
        return fig, ax


class DynamicVisualizer():
    """Class for visualizing dynamic network features.
    
    Parameters
    ----------
    mask_file : str
        Path to the brain mask file.
    parcellation_file : str
        Path to the parcellation file used during the data source reconstruction.
    """
    def __init__(self):
        self.mask_file="MNI152_T1_8mm_brain.nii.gz"
        self.parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

    def plot_power_map(
        self,
        power_map,
        filename,
        subtract_mean=False,
        mean_weights=None,
        colormap=None,
        fontsize=20,
        plot_kwargs={},
    ):
        """Plots state-specific power map(s). Wrapper for `osl_dynamics.analysis.power.save().

        Parameters
        ----------
        power_map : np.ndarray
            Power map to save. Can be of shape: (n_components, n_modes, n_channels),
            (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
            array can also be passed. Warning: this function cannot be used if n_modes
            is equal to n_channels.
        filename : str
            File name to be used when saving a figure object.
        subtract_mean : bool
            Should we subtract the mean power across modes? Defaults to False.
        mean_weights : np.ndarray
            Numpy array with weightings for each mode to use to calculate the mean.
            Default is equal weighting.
        colormap : str
            Colors for connectivity edges. If None, a default colormap is used 
            ("cold_hot").
        fontsize : int
            Fontsize for a power map colorbar. Defaults to 20.
        plot_kwargs : dict
            Keyword arguments to pass to `nilearn.plotting.plot_img_on_surf`.
        """
        # Set visualization parameters
        if colormap is None:
            colormap = "cold_hot"

        # Plot surface power maps
        plot_kwargs.update({
            "cmap": colormap,
            "views": ["lateral", "medial"],
        })
        figures, axes = power.save(
            power_map=power_map,
            mask_file=self.mask_file,
            parcellation_file=self.parcellation_file,
            subtract_mean=subtract_mean,
            mean_weights=mean_weights,
            plot_kwargs=plot_kwargs,
        )
        for i, fig in enumerate(figures):
            # Reset figure size
            fig.set_size_inches(5, 6)
            
            # Change colorbar position
            cb_ax = fig.axes[-1]
            pos = cb_ax.get_position()
            new_pos = [pos.x0 * 0.92, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
            cb_ax.set_position(new_pos)
            
            # Set colorbar styles
            _format_colorbar_ticks(cb_ax)
            cb_ax.xaxis.set_major_formatter(ScalarFormatter())
            cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 6))
            cb_ax.tick_params(labelsize=fontsize)
            cb_ax.xaxis.offsetText.set_fontsize(fontsize)
            if len(figures) > 1:
                tmp_filename = filename.replace(
                    filename.split('.')[0], filename.split('.')[0] + f"_{i}"
                )
                save(fig, tmp_filename, transparent=True)
            else:
                save(fig, filename, transparent=True)

        return None

    def plot_coh_conn_map(
        self,
        connectivity_map,
        filename,
        colormap="bwr",
        plot_kwargs={},
    ):
        """Plots state-specific connectivity map(s). Wrapper for `osl_dynamics.analysis.connectivity.save()`.

        Parameters
        ----------
        connectivity_map : np.ndarray
            Matrices containing connectivity strengths to plot. Shape must be 
            (n_modes, n_channels, n_channels) or (n_channels, n_channels).
        filename : str
            File name to be used when saving a figure object.
        colormap : str
            Type of a colormap to use for connectivity edges. Defaults to "bwr".
        plot_kwargs : dict
            Keyword arguments to pass to `nilearn.plotting.plot_connectome`.
        """
        # Validation
        if connectivity_map.ndim == 2:
            connectivity_map = connectivity_map[np.newaxis, ...]

        # Number of states/modes
        n_states = connectivity_map.shape[0]

        # Plot connectivity maps
        for n in range(n_states):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
            temp_kwargs = {"edge_cmap": colormap, "figure": fig}
            connectivity.save(
                connectivity_map=connectivity_map[n, :],
                parcellation_file=self.parcellation_file,
                axes=[ax],
                plot_kwargs={**plot_kwargs, **temp_kwargs},
            )
            cb_ax = fig.get_axes()[-1]
            pos = cb_ax.get_position()
            new_pos = [pos.x0 * 1.05, pos.y0, pos.width, pos.height]
            cb_ax.set_position(new_pos)
            cb_ax.tick_params(labelsize=20)
            if n_states != 1:
                tmp_filename = filename.replace(
                    filename.split('.')[0], filename.split('.')[0] + f"_{n}"
                )
                save(fig, tmp_filename, transparent=True)
            else:
                save(fig, filename, transparent=True)

        return None
    
    def plot_psd(
        self,
        freqs,
        psd,
        mean_psd,
        filename,
        fontsize=22,
    ):
        """Plots state-specific subject-averaged PSD(s).

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequencies in the frequency axis. Shape is (n_freqs,).
        psd : np.ndarray
            State-specific subject-averaged PSD. Shape must be (n_states, n_freqs).
        mean_psd : np.ndarray
            Subject-averaged PSD, averaged across all states/modes.
            Shape must be ( n_freqs,).
        filename : str
            File name to be used when saving a figure object.
        fontsize : int
            Fontsize for axes ticks and labels. Defaults to 22.
        """
        # Validation
        if (psd.ndim != 2) or (mean_psd.ndim != 1):
            raise ValueError("psd and mean_psd should be 2-D and 1-D arrays, respectively.")
        
        # Number of states
        n_states = psd.shape[0]

        # Plot PSDs for each state and their mean across states/modes
        hmin, hmax = 0, np.ceil(freqs[-1])
        vmin = min([np.min(psd), np.min(mean_psd)])
        vmax = max([np.max(psd), np.max(mean_psd)])
        vspace = 0.1 * (vmax - vmin)

        for n in range(n_states):
            fig, ax = plot_line(
                [freqs],
                [psd[n]],
                plot_kwargs={"lw": 1.5},
            )
            ax.plot(freqs, mean_psd, color="black", linestyle="--", lw=1.5)

            # Set axis labels
            ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
            ax.set_ylabel("PSD (a.u.)", fontsize=fontsize)

            # Reset figure size
            fig.set_size_inches(6, 4)
            
            # Set axis styles
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(2)
            ax.set_xticks(np.arange(hmin, hmax, 10))
            ax.set_ylim([vmin - vspace, vmax + vspace])
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
            ax.tick_params(labelsize=fontsize)
            ax.yaxis.offsetText.set_fontsize(fontsize)
            plt.tight_layout()

            # Save figure
            if n_states != 1:
                tmp_filename = filename.replace(
                    filename.split('.')[0], filename.split('.')[0] + f"_{n}"
                )
                save(fig, tmp_filename, transparent=True)
            else:
                save(fig, filename, transparent=True)

        return None

    def plot_loss(
        self,
        history,
        model_type,
        save_dir,
        epoch_step=None,
    ):
        """Plots training loss curves.
        
        Parameters
        ----------
        history : dict
            Training history containing loss values.
        model_type : str
            Type of the model used. Should be either "hmm" or "dyneste".
        save_dir : str
            Directory to save the loss plot(s).
        epoch_step : int, optional
            Step size for x-axis ticks. If None, defaults to 5.
        """
        # Load loss values
        loss = history["loss"]
        if model_type == "dyneste":
            ll_loss = history["ll_loss"]
            kl_loss = history["kl_loss"]
        
        # Set epochs
        epochs = np.arange(1, len(loss) + 1)
        if epoch_step is None:
            epoch_step = 5

        # Plot training loss curve
        fig, ax = plot_line(
            [epochs],
            [loss],
            plot_kwargs={"lw": 2},
        )
        ax.set_xticks(np.arange(0, len(loss) + epoch_step, epoch_step))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(axis="both", which="both", labelsize=18, width=2)
        ax.set_xlabel("Epochs", fontsize=18)
        ax.set_ylabel("Loss", fontsize=18)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss.png"), transparent=True)
        plt.close(fig)

        # Plot loss sub-components
        if model_type == "dyneste":
            # Plot loss components
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
            ax[0].plot(epochs, ll_loss, color="tab:red", lw=3)
            ax[1].plot(epochs, kl_loss, color="tab:green", lw=3)
            for i in range(2):
                ax[i].set_xlabel("Epochs", fontsize=20)
                ax[i].set_xticks([0, len(epochs)])
                ax[i].yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
                ax[i].tick_params(axis='both', which='both', labelsize=20, width=3)
                for axis in ["top", "bottom", "left", "right"]:
                    ax[i].spines[axis].set_linewidth(3)
            ax[0].set_ylabel("Loss", fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "loss_ll_kl.png"))
            plt.close(fig)

        return None
