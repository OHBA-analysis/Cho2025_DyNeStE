"""Functions for data visualization and plotting.

This module contains several functions from osl_dynamics.utils.plotting.py,
which has been edited for the purpose of this project.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils.plotting import plot_line, plot_matrices
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


def plot_statewise_matrix(
    matrix,
    mark_diagonal=True,
    colormap="magma",
    fontsize=14,
    axis_labels=[],
    cbar_label="",
    filename=None,
):
    """Plots a state-wise matrix.

    Parameters
    ----------
    matrix : np.ndarray
        State-wise matrix. Shape must be (n_states, n_states).
    mark_diagonal : bool, optional
        Whether to insert texts on the diagonal. Defaults to True.
    colormap : str or matplotlib.colors.ListedColormap, optional
        Colormap to use for the matrix. Defaults to "magma".
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 14.
    axis_labels : list of str, optional
        List of labels for the x and y axes. If not provided, the
        default labels will be "States".
    cbar_label : str, optional
        Label for the colorbar.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    axes : plt.axes
        Matplotlib axis object(s). Only returned if :code:`filename=None`.
    """
    # Validation
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be a 2D array.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be a square matrix.")
    if axis_labels:
        xlbl = axis_labels[0]
        ylbl = axis_labels[1]
    else:
        xlbl, ylbl = "States", "States"

    # Get the number of states
    n_states = matrix.shape[0]

    # Visualize the correlation matrix
    fig, axes = plot_matrices(matrix, cmap=colormap)
    ax = axes[0][0]
    if mark_diagonal:
        for (i, j), val in np.ndenumerate(matrix):
            if i == j:
                ax.text(
                    j, i, "{:.2f}".format(val), ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3"),
                    fontsize=9,
                )
    ticks = np.arange(0, n_states, 2)
    tick_labels = ticks + 1
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel(xlbl, fontsize=fontsize)
    ax.set_ylabel(ylbl, fontsize=fontsize)

    # Set tick placement and appearance
    ax.tick_params(labelsize=fontsize)
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(axis="y", left=True, labeltop=True, right=False, labelright=False)

    # Adjust colorbar
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel(cbar_label, fontsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize)
    
    # Adjust colorbar location
    pos = cbar_ax.get_position()
    new_pos = [pos.x0 * 0.95, pos.y0, pos.width, pos.height]
    cbar_ax.set_position(new_pos)

    # Save or return the figure
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, axes
    

def plot_violin(
    df,
    x_var,
    y_var,
    hue_var,
    palette,
    ylim=None,
    figsize=None,
    fontsize=14,
    filename=None,
):
    """Plots a violin plot using seaborn.

    This function supports a grouped violin plot as well.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be plotted.
    x_var : str
        Column name in the DataFrame for the x-axis variable.
    y_var : str
        Column name in the DataFrame for the y-axis variable.
    hue_var : str
        Column name in the DataFrame for the hue variable.
    palette : dict
        Dictionary mapping hue variable values to colors.
    ylim : list, optional
        List of two floats specifying the y-axis limits.
    figsize : tuple, optional
        Tuple specifying the figure size (width, height).
        Defaults to (5, 1).
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
    # Validate inputs
    if figsize is None:
        figsize = (5, 1)

    # Visualize violin plot
    sns.set_theme(style="white")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.violinplot(
        data=df,
        x=x_var,
        y=y_var,
        hue=hue_var,
        palette=palette,
        inner="box",
        density_norm="count",
        legend="auto",
        linewidth=1,
        ax=ax,
    )

    # Adjust the legend
    lgnd_fontsize = fontsize - 3
    legend = ax.legend(fontsize=lgnd_fontsize, loc="upper left")
    legend.set_title(hue_var, prop={"size": lgnd_fontsize})

    # Adjust tick settings
    ticks = np.arange(df[x_var].nunique())
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.tick_params(labelsize=fontsize)

    # Adjust axis settings
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(x_var, fontsize=fontsize)
    ax.set_ylabel(y_var, fontsize=fontsize)
    ax.spines[["left", "bottom", "right", "top"]].set_linewidth(2)
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.13, top=0.93)  # fix padding for constrained layout
    if filename is not None:
        fig.savefig(filename, dpi=300)
        plt.close(fig)
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
    """Plots Fano factors for inferred and sampled state time courses.

    Parameters
    ----------
    fano_factors : list of np.ndarray
        List of Fano factors for inferred and sampled state time courses.
        Each element should be a 2-D or 3-D array. Shape must be either 
        (n_windows, n_states) or (n_subjects, n_windows, n_states).
    window_lengths : np.ndarray
        Array of window lengths used for Fano factor calculation.
        Shape must be (n_windows,).
    sampling_frequency : int
        Sampling frequency.
    sig_indices : list of lists, optional
        List of indices for significant Fano factors for inferred and sampled 
        state time courses. If provided, the corresponding Fano factors will be
        highlighted. Defaults to None.
    ylims : list, optional
        List of two floats specifying the y-axis limits for each subplot.
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
    for n, fano in enumerate(fano_factors):
        if fano.ndim == 3:
            fano = np.mean(fano, axis=0)  # average over subjects
            fano_factors[n] = fano
            # shape: (n_windows, n_states)
        if fano.ndim != 2:
            raise ValueError("Fano factors should be either a 2-D or 3-D array.")
        if fano.shape[0] != len(window_lengths):
            raise ValueError(
                "Fano factors should have the same number of windows as window_lengths."
            )
        
    if sig_indices is not None:
        if len(sig_indices) != 2:
            raise ValueError(
                "there should be sig_indices for both inferred and sampled data."
            )
    
    # Get user inputs
    inf_fano_factors = fano_factors[0]
    sam_fano_factors = fano_factors[1]

    if ylims is None:
        min_fano, max_fano = np.min(fano_factors), np.max(fano_factors)
        gap = (max_fano - min_fano) * 0.1
        ylims = [
            [min_fano - gap, max_fano + gap],
            [min_fano - gap, max_fano + gap],
        ]

    # Get the number of states
    if inf_fano_factors.shape[1] != sam_fano_factors.shape[1]:
        raise ValueError("Fano factors should have the same number of states.")
    n_states = fano_factors[0].shape[1]

    # Get the significant indices if provided
    inf_idx, sam_idx = None, None
    if sig_indices is not None:
        inf_idx = sig_indices[0]
        sam_idx = sig_indices[1]

    # Set colormap
    palette, color_ls, _ = _get_color_tools("tol_bright", n_states)
    
    # Visualize Fano factors
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
    x_windows = (window_lengths / sampling_frequency) * 1000 # samples to milliseconds
    for i in range(n_states):
        ax[0].plot(
            x_windows, inf_fano_factors[:, i],
            linestyle=color_ls[i], color=palette[i],
        )
        ax[1].plot(
            x_windows, sam_fano_factors[:, i],
            linestyle=color_ls[i], color=palette[i],
        )
        
    if inf_idx:
        y_lower = np.ones(len(x_windows)) * ylims[0][0]
        y_upper = np.ones(len(x_windows)) * ylims[0][1]
        for idx in inf_idx:
            ax[0].fill_between(
                x_windows[idx],
                y_lower[idx],
                y_upper[idx],
                color="tab:green",
                edgecolor="none",
                alpha=0.1,
            )
    if sam_idx:
        y_lower = np.ones(len(x_windows)) * ylims[1][0]
        y_upper = np.ones(len(x_windows)) * ylims[1][1]
        for idx in sam_idx:
            ax[1].fill_between(
                x_windows[idx],
                y_lower[idx],
                y_upper[idx],
                color="tab:red",
                edgecolor="none",
                alpha=0.1,
            )
    
    for a in range(2):  # iterate over axes
        ax[a].set_xscale("log")
        ax[a].tick_params(labelsize=fontsize)
        ax[a].set_xlabel("Window Length (ms)", fontsize=fontsize)
        ax[a].set_ylim(ylims[a])
    ax[0].set_ylabel("Fano Factor", fontsize=fontsize)
    ax[0].set_title("Inferred", fontsize=fontsize)
    ax[1].set_title("Sampled", fontsize=fontsize)
    plt.tight_layout()
    if filename is not None:
        save(fig, filename)
    else:
        return fig, ax


def plot_mutual_information(
    mutual_information,
    lags,
    sampling_frequency,
    sig_indices=None,
    xticks=None,
    ylims=None,
    fontsize=14,
    filename=None,
):
    """Plots mutual information between a state and other states across lags.

    Parameters
    ----------
    mutual_information : list of np.ndarray
        List of mutual information arrays for inferred and sampled state time
        courses. Shape of each array must be either (n_lags, n_states) or
        (n_subjects, n_lags, n_states).
    lags : list or np.ndarray
        Array of lags used for mutual information calculation.
        Shape must be (n_lags,).
    sampling_frequency : int
        Sampling frequency.
    sig_indices : list of lists, optional
        List of indices for significant Fano factors for inferred and sampled 
        state time courses. If provided, the corresponding Fano factors will be
        highlighted. Defaults to None.
    xticks : np.ndarray, optional
        Array of x-axis tick values. If None, defaults to None.
    ylims : list, optional
        List of two floats specifying the y-axis limits for each subplot.
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
        Matplotlib axis object(s). Only returned if :code:`filename=None`.
    """
    # Validation
    for n, mi in enumerate(mutual_information):
        if mi.ndim == 3:
            mi = np.mean(mi, axis=0) # average over subjects
            mutual_information[n] = mi
            # shape: (n_lags, n_states)
        if mi.ndim != 2:
            raise ValueError("mutual information should be a 3-D array.")
        if mi.shape[0] != len(lags):
            raise ValueError(
                "mutual information should have the same number of lags as input lags."
            )
        
    if sig_indices is not None:
        if len(sig_indices) != 2:
            raise ValueError(
                "there should be sig_indices for both inferred and sampled data."
            )
    
    # Get user inputs
    inf_mi = mutual_information[0]
    sam_mi = mutual_information[1]

    if ylims is None:
        min_mi = np.min(mutual_information)
        max_mi = np.max(mutual_information)
        gap = (max_mi - min_mi) * 0.1
        ylims = [
            [min_mi - gap, max_mi + gap],
            [min_mi - gap, max_mi + gap],
        ]

    # Get the number of states
    if inf_mi.shape[-1] != sam_mi.shape[-1]:
        raise ValueError("mutual information should have the same number of states.")
    n_states = inf_mi.shape[-1]

    # Get the significant indices if provided
    inf_idx, sam_idx = None, None
    if sig_indices is not None:
        inf_idx = sig_indices[0]
        sam_idx = sig_indices[1]

    # Set colormap
    palette, color_ls, color_mk = _get_color_tools("tol_bright", n_states)

    # Convert samples to seconds
    x_lags = lags / sampling_frequency
    xticks = xticks / sampling_frequency if xticks is not None else None

    # Visualize mutual information
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    for i in range(n_states):
        ax[0].plot(
            x_lags, inf_mi[:, i], marker=color_mk[i],
            ls=color_ls[i], color=palette[i], alpha=0.7,
        )
        ax[1].plot(
            x_lags, sam_mi[:, i], marker=color_mk[i],
            ls=color_ls[i], color=palette[i], alpha=0.7,
        )
    
    if inf_idx:
        y_lower = np.ones(len(x_lags)) * ylims[0][0]
        y_upper = np.ones(len(x_lags)) * ylims[0][1]
        for idx in inf_idx:
            ax[0].fill_between(
                x_lags[idx],
                y_lower[idx],
                y_upper[idx],
                color="tab:green",
                edgecolor="none",
                alpha=0.1,
            )
    if sam_idx:
        y_lower = np.ones(len(x_lags)) * ylims[1][0]
        y_upper = np.ones(len(x_lags)) * ylims[1][1]
        for idx in sam_idx:
            ax[1].fill_between(
                x_lags[idx],
                y_lower[idx],
                y_upper[idx],
                color="tab:red",
                edgecolor="none",
                alpha=0.1,
            )

    for a in range(2):  # iterate over axes
        if xticks is not None:
            ax[a].set_xticks(xticks)
        ax[a].ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
        ax[a].tick_params(labelsize=fontsize)
        ax[a].yaxis.offsetText.set_fontsize(fontsize)
        ax[a].set_xlabel("Lags (s)", fontsize=fontsize)
        ax[a].set_ylim(ylims[a])
    ax[0].set_ylabel("Average Mutual Information (normalized)", fontsize=fontsize)
    ax[0].set_title("Inferred", fontsize=fontsize)
    ax[1].set_title("Sampled", fontsize=fontsize)
    plt.tight_layout()
    if filename is not None:
        save(fig, filename)
    else:
        return fig, ax
    

def plot_asymmetry_matrix(
    matrix,
    state_sequence,
    edge_idx=None,
    vlims=None,
    fontsize=16,
    filename=None,
):
    """Plots an asymmetry matrix from the TINDA analysis.

    Parameters
    ----------
    matrix : np.ndarray
        Asymmetry matrix to plot. Shape must be (n_states, n_states) or
        (n_states, n_states, n_subjects).
    state_sequence : np.ndarray
        Sequence of states to reorder the matrix. Shape must be (n_states,).
    edge_idx : np.ndarray
        Indices of the edges to mark. Shape is (n_edges, 2).
    vlims : list, optional
        List of two floats specifying the color limits for the matrix.
        Defaults to None, in which case the limits are determined automatically.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 16.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object. Only returned if :code:`filename=None`.
    """
    # Vadlidate inputs
    if matrix.ndim == 3:
        matrix = np.nanmean(matrix, axis=2)  # average over subjects
    elif matrix.ndim != 2:
        raise ValueError("Input matrix must be a 2D or 3D array.")
    
    # Reorder matrix by state sequence
    state_sequence = np.concatenate((
        [state_sequence[0]], state_sequence[1:][::-1]
    ))  # change from counter-clockwise to clockwise
    matrix = matrix[np.ix_(state_sequence, state_sequence)]

    # Plot asymmetry matrix
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    if vlims is None:
        img = ax.imshow(matrix, cmap="RdBu_r")
    else:
        img = ax.imshow(matrix, cmap="RdBu_r", vmin=vlims[0], vmax=vlims[1])

    if edge_idx is not None:
        for i, j in edge_idx:
            ax.text(j, i, "*", ha="center", va="center",
                    color="black", fontsize=fontsize, fontweight="bold")
            
    # Adjust axis settings
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(np.array(state_sequence) + 1)
    ax.set_yticklabels(np.array(state_sequence) + 1)
    ax.set_xlabel("Reference State (n)", fontsize=fontsize)
    ax.set_ylabel("Network State (m)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(img, cax=cax)
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    cbar.ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.offsetText.set_fontsize(fontsize)

    # Save or return the figure
    plt.tight_layout()
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, ax
    

def _plot_cycle(
    ordering,
    fo_density,
    edges,
    fontsize=16,
):
    """Plot state network as circular diagram with arrows.

    Parameters
    ----------
    ordering : list
        List of best sequence of states to plot (in order of counterclockwise
        rotation).
    fo_density : array_like
        Time-in-state densities array of shape (n_interval_states,
        n_density_states, 2, (n_interval_ranges,) n_sessions).
    edges : array_like
        Array of zeros and ones indicating whether the connection should be
        plotted.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 16.
    """
    # Set colormap
    palette, _, _ = _get_color_tools("tol_bright", len(ordering))
    colormap = palette.copy()
    colormap = [matplotlib.colors.to_rgba(clr) for clr in colormap]
    colormap = [clr[:3] + (0.8,) for clr in colormap]  # set alpha to 0.8

    # Plot state network as circular diagram with arrows
    plt.gca()

    K = len(ordering)
    if len(fo_density.shape) == 5:
        fo_density = np.squeeze(
            fo_density
        )  # squeeze in case there is still a interval_ranges dimension

    # Compute mean direction of arrows
    mean_direction = np.squeeze(
        (fo_density[:, :, 0, :] - fo_density[:, :, 1, :]).mean(axis=2)
    )  # equivalent to asymmetry matrix

    # Reorder the states to match the ordering
    ordering = np.roll(ordering[::-1], 1)
    # rotate ordering from clockwise to counter clockwise
    edges = edges[ordering][:, ordering]
    mean_direction = mean_direction[ordering][:, ordering]

    # Get the locations on the unit circle
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / K)
    x = np.roll(np.cos(theta), int(K / 4))  # start from 12 o'clock
    y = np.roll(np.sin(theta), int(K - (K / 4)))
    distance_to_plot_manual = np.stack([x, y]).T

    # Plot the scatter points with state identities
    for i in range(K):
        plt.scatter(
            distance_to_plot_manual[i, 0],
            distance_to_plot_manual[i, 1],
            s=1000,
            color=colormap[ordering[i]],
        )
        plt.text(
            distance_to_plot_manual[i, 0],
            distance_to_plot_manual[i, 1],
            str(ordering[i] + 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fontsize,
            fontweight="bold",
        )

    # Plot the arrows
    for ik1 in range(K):
        for k2 in range(K):
            if edges[ik1, k2]:
                # arrow lengths have to be proportional to the distance
                # between the states. Use Pythagoras.
                line_scale = np.sqrt(
                    np.sum(
                        (
                            distance_to_plot_manual[k2, :]
                            - distance_to_plot_manual[ik1, :]
                        )
                        ** 2
                    )
                )
                arrow_start = (
                    distance_to_plot_manual[ik1, :]
                    + 0.1
                    * (distance_to_plot_manual[k2, :] - distance_to_plot_manual[ik1, :])
                    / line_scale
                )
                arrow_end = (
                    distance_to_plot_manual[k2, :]
                    - 0.1
                    * (distance_to_plot_manual[k2, :] - distance_to_plot_manual[ik1, :])
                    / line_scale
                )
                if mean_direction[ik1, k2] > 0:  # arrow from k1 to k2
                    plt.arrow(
                        arrow_start[0],
                        arrow_start[1],
                        arrow_end[0] - arrow_start[0],
                        arrow_end[1] - arrow_start[1],
                        head_width=0.05,
                        head_length=0.1,
                        length_includes_head=True,
                        color="k",
                    )
                elif mean_direction[ik1, k2] < 0:  # arrow from k2 to k1
                    plt.arrow(
                        arrow_end[0],
                        arrow_end[1],
                        arrow_start[0] - arrow_end[0],
                        arrow_start[1] - arrow_end[1],
                        head_width=0.05,
                        head_length=0.1,
                        length_includes_head=True,
                        color="k",
                    )
    plt.axis("off")
    plt.axis("equal")


def plot_tinda_cycle(
    fo_density,
    state_sequence,
    edges,
    fontsize=16,
    filename=None,
):
    """Plots a state cycle computed using the TINDA algorithm.
       Wrapper for :func:`_plot_cycle`.

    Parameters
    ----------
    fo_density : np.ndarray
        Fractional occupancy density matrix.
        Shape is (n_interval_states, n_density_states, n_bins,
        n_interval_ranges, n_subjects).
    state_sequence : np.ndarray
        Sequence of states.
    edges : np.ndarray
        Binary matrix indicating the presence of edges.
        Shape is (n_states, n_states).
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 16.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object. Only returned if :code:`filename=None`.
    """
    # Plot TINDA cycle
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    _plot_cycle(state_sequence, fo_density, edges, fontsize=fontsize)

    # Save or return the figure
    plt.tight_layout()
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, ax


def plot_cycle_strengths(
    df,
    palette,
    p_vals=None,
    fontsize=14,
    filename=None,
):
    """Plots cycle strengths from the DyNeStE and HMM models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cycle strengths data.
    palette : dict
        Dictionary mapping hue variable values to colors.
    p_vals : list of float
        List of p-values of statistical tests on each correspoding group 
        of cycle strengths. Defaults to None.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 14.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object. Only returned if :code:`filename=None`.
    """
    # Set group orders
    order = ["Inferred", "Sampled"]
    hue_order = ["DyNeStE", "HMM"]

    # Plot boxplots 
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5.5))
    bp = sns.boxplot(
        data=df, x="Data Type", y="Cycle Strengths",
        hue="Models", palette=palette, width=0.6, gap=0.1,
        order=order, hue_order=hue_order,
        showmeans=True, meanprops={
            "marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"
        },
        fill=False, legend=False, zorder=2, ax=ax,
    )

    # Visualize manual strip plots
    for i, container in enumerate(bp.containers):
        for j, box in enumerate(container.boxes):
            y = df[
                (df["Data Type"] == order[j]) & (df["Models"] == hue_order[i])
            ]["Cycle Strengths"].values
            x_data = box.get_xdata()
            x_center = (min(x_data) + max(x_data)) / 2
            x = np.random.normal(x_center, 0.02, size=len(y))
            ax.scatter(
                x, y, color=palette[hue_order[i]], edgecolor="none", 
                marker="o", alpha=0.3, zorder=1,
            )

    # Add significance labels
    if p_vals is not None:
        for i, p in enumerate(p_vals):
            p_label = _categorize_pvalue(p)
            ax.text(
                i, ax.get_ylim()[1] * 0.98, p_label,
                color="k", ha="center", va="bottom",
                fontweight="bold", fontsize=fontsize,
            )

    # Adjust axis settings
    vmin, vmax = ax.get_ylim()
    gap = (vmax - vmin) * 0.05
    ax.set_ylim([vmin, vmax + gap])
    ax.set_xlabel("Data Type", fontsize=fontsize)
    ax.set_ylabel("Cycle Strength (a.u.)", fontsize=fontsize)
    ax.tick_params(axis="both", which="both", width=1.5, labelsize=fontsize)
    ax.spines[["left", "bottom", "right", "top"]].set_linewidth(1.5)
    
    # Save or return the figure
    plt.tight_layout()
    if filename is not None:
        save(fig, filename, transparent=True)
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
