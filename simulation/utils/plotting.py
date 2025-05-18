"""Functions for data visualization.

This module contains several functions from osl_dynamics.utils.plotting.py,
which has been edited for the purpose of this project.
"""

import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from itertools import zip_longest
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import gamma

from osl_dynamics.array_ops import get_one_hot
from osl_dynamics.analysis import modes
from osl_dynamics.utils import plotting as osld_plotting
from osl_dynamics.utils.misc import override_dict_defaults
from osl_dynamics.utils.plotting import create_figure, rough_square_axes


# Suppress matplotlib warnings
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)


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


def plot_alpha(
    *alpha,
    n_samples=None,
    colors="tab10",
    sampling_frequency=None,
    y_labels=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    filename=None,
    axes=None,
):
    """Plots alpha.

    Parameters
    ----------
    alpha : np.ndarray
        A collection of alphas passed as separate arguments.
    n_samples: int, optional
        Number of time points to be plotted.
    colors : list or str, optional
        List of colors for the colormap, or a matplotlib colormap string.
        Defaults to :code:`"tab10"`.
    sampling_frequency : float, optional
        The sampling frequency of the data in Hz.
    y_labels : str, optional
        Labels for the y-axis of each alpha time series.
    title : str, optional
        Title for the plot.
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
    axes : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """
    n_alphas = len(alpha)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    if axes is not None and len(axes) != n_alphas:
        raise ValueError("Number of axes must match number of alphas.")

    n_states = max(a.shape[1] for a in alpha)
    n_samples = min(n_samples or np.inf, alpha[0].shape[0])

    # Create a colormap
    if isinstance(colors, str):
        cmap = plt.cm.get_cmap(name=cmap)
    else:
        cmap = matplotlib.colors.ListedColormap(colors)
    colors = cmap.colors

    if "alpha" in plot_kwargs:
        rgba_colors = [
            matplotlib.colors.to_rgba(c, alpha=plot_kwargs["alpha"]) for c in colors
        ]
        cmap = matplotlib.colors.ListedColormap(rgba_colors)
        colors = cmap.colors

    if len(colors) != n_states:
        raise ValueError(f"Number of colors must match the number of states.")

    # Validation
    if fig_kwargs is None:
        fig_kwargs = {}
    default_fig_kwargs = dict(
        figsize=(14, 2.5 * n_alphas), sharex="all", facecolor="white"
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
        ax.set_ylabel(y_label)

    # Set axis label and title
    axes[-1].set_xlabel("Time (s)" if sampling_frequency else "Sample")
    axes[0].set_title(title)

    # Fix layout
    plt.tight_layout()

    # Add a colour bar
    norm = matplotlib.colors.BoundaryNorm(
        boundaries=range(n_states + 1), ncolors=n_states
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(right=0.94)
    cb_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    cb = fig.colorbar(mappable, cax=cb_ax, ticks=np.arange(0.5, n_states, 1))
    cb.ax.set_yticklabels(range(1, n_states + 1))

    # Save to file if a filename as been passed
    if filename is not None:
        save(fig, filename)
    else:
        return fig, axes


def plot_matrices(
    matrix,
    group_color_scale=True,
    titles=None,
    main_title=None,
    cmap="viridis",
    nan_color="white",
    cbar_label="",
    log_norm=False,
    filename=None,
):
    """Plots a collection of matrices.

    Given an iterable of matrices, plot each matrix in its own axis. The axes
    are arranged as close to a square (:code:`N x N` axis grid) as possible.

    Parameters
    ----------
    matrix: list of np.ndarray
        The matrices to plot.
    group_color_scale: bool, optional
        If True, all matrices will have the same colormap scale, where we use
        the minimum and maximum across all matrices as the scale.
    titles: list of str, optional
        Titles to give to each matrix axis.
    main_title: str, optional
        Main title to be placed at the top of the plot.
    cmap: str, optional
        Matplotlib colormap.
    nan_color: str, optional
        Matplotlib color to use for :code:`NaN` values.
    cbar_label : str, optional
        Label for the colorbar.
    log_norm: bool, optional
        Should we show the elements on a log scale?
    filename: str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    axes : plt.axes
        Matplotlib axis object(s). Only returned if :code:`filename=None`.
    """
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        matrix = matrix[None, :]
    if matrix.ndim != 3:
        raise ValueError("Must be a 3D array.")
    short, long, _ = rough_square_axes(len(matrix))
    fig, axes = plt.subplots(ncols=long, nrows=short, figsize=(2.5 * long, 2.5 * short))

    if titles is None:
        titles = [""] * len(matrix)

    cmap = matplotlib.cm.get_cmap(cmap).copy()
    cmap.set_bad(color=nan_color)

    for i, (grid, axis, title) in enumerate(zip_longest(matrix, axes.ravel(), titles)):
        if grid is None:
            axis.remove()
            continue
        if group_color_scale:
            v_min = np.nanmin(matrix)
            v_max = np.nanmax(matrix)
            if log_norm:
                im = axis.matshow(
                    grid,
                    cmap=cmap,
                    norm=matplotlib.colors.LogNorm(vmin=v_min, vmax=v_max),
                )
            else:
                im = axis.matshow(grid, vmin=v_min, vmax=v_max, cmap=cmap)
        else:
            if log_norm:
                im = axis.matshow(
                    grid,
                    cmap=cmap,
                    norm=matplotlib.colors.LogNorm(),
                )
            else:
                im = axis.matshow(grid, cmap=cmap)
        axis.set_title(title)
        axis.tick_params(
            axis="both",
            which="both",
            top=False,
            bottom=True,
            left=True,
            right=False,
            labeltop=False,
            labelbottom=True,
            labelleft=True,
            labelright=False,
        )
        if grid.shape[0] > 30:
            # Don't label the ticks if there's too many
            axis.set_xticklabels([])
            axis.set_yticklabels([])
        else:
            ticks = np.arange(grid.shape[0], step=2)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            # Set ytick labels only for the first column
            if i % long == 0:
                axis.set_yticklabels(ticks + 1)
                axis.set_ylabel("Channels", fontsize=12)
            else:
                axis.set_yticklabels([])
            # Set xtick labels only for the last row
            if i >= len(matrix) - long:
                axis.set_xticklabels(ticks + 1)
                axis.set_xlabel("Channels", fontsize=12)
            else:
                axis.set_xticklabels([])
            axis.tick_params(labelsize=12)

    if group_color_scale:
        fig.subplots_adjust(right=0.8, wspace=0.2, hspace=0.01)
        color_bar_axis = fig.add_axes([0.85, 0.23, 0.04, 0.5])
        color_bar = fig.colorbar(im, cax=color_bar_axis)
        color_bar.set_label(cbar_label, fontsize=12)
    else:
        for axis in fig.axes:
            pl = axis.get_images()[0]
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            color_bar = plt.colorbar(pl, cax=cax)
            color_bar.set_label(cbar_label, fontsize=12)
        plt.tight_layout()

    fig.suptitle(main_title)

    if filename is not None:
        save(fig, filename)
    else:
        return fig, axes


def _plot_state_lifetimes(
    state_time_course,
    bins="auto",
    density=False,
    colors=None,
    match_scale_x=False,
    match_scale_y=False,
    x_range=None,
    x_label=None,
    y_label=None,
    plot_kwargs=None,
    fig_kwargs=None,
    filename=None,
):
    """Creates a histogram of state lifetimes.

    For a state time course, create a histogram for each state with the
    distribution of the lengths of time for which it is active.

    Parameters
    ----------
    state_time_course : np.ndarray
        State time course to analyze.
    bins : int, optional
        Number of bins for the histograms.
    density : bool, optional
        If :code:`True`, plot the probability density of the state activation
        lengths. If :code:`False`, raw number.
    colors : list, optional
        List of colors for the histograms. If :code:`None`, a default
        colormap is used.
    match_scale_x : bool, optional
        If True, all histograms will share the same x-axis scale.
    match_scale_y : bool, optional
        If True, all histograms will share the same y-axis scale.
    x_range : list, optional
        The limits on the values presented on the x-axis.
    x_label : str, optional
        x-axis label.
    y_label : str, optional
        y-axis label.
    plot_kwargs : dict, optional
        Keyword arguments to pass to `ax.hist <https://matplotlib.org/stable\
        /api/_as_gen/matplotlib.axes.Axes.hist.html>`_.
    fig_kwargs : dict, optional
        Arguments to pass to :code:`plt.subplots()`.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`ax=None` and
        :code:`filename=None`.
    axes : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """
    n_plots = state_time_course.shape[1]
    short, long, _ = rough_square_axes(n_plots)
    if colors is None:
        colors = osld_plotting.get_colors(n_plots)

    # Validation
    if state_time_course.ndim == 1:
        state_time_course = get_one_hot(state_time_course)
    if state_time_course.ndim != 2:
        raise ValueError("state_time_course must be a 2D array.")

    if fig_kwargs is None:
        fig_kwargs = {}
    default_fig_kwargs = {"figsize": (long * 2.5, short * 2.5)}
    fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    # Calculate state lifetimes
    channel_lifetimes = modes.lifetimes(state_time_course)

    # Create figure
    fig, axes = create_figure(short, long, **fig_kwargs)

    # Plot data
    largest_bar = 0
    furthest_value = 0
    for channel, axis, color in zip_longest(channel_lifetimes, axes.ravel(), colors):
        if channel is None:
            axis.remove()
            continue
        if not len(channel):
            axis.text(
                0.5,
                0.5,
                "No\nactivation",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
                fontsize=20,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            continue
        hist = axis.hist(
            channel, density=density, bins=bins, color=color, **plot_kwargs
        )
        largest_bar = max(hist[0].max(), largest_bar)
        furthest_value = max(hist[1].max(), furthest_value)
        t = axis.text(
            0.95,
            0.95,
            f"{np.sum(channel) / len(state_time_course) * 100:.2f}%",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="top",
            transform=axis.transAxes,
        )
        axis.xaxis.set_tick_params(labelbottom=True, labelleft=True)
        t.set_bbox({"facecolor": "white", "alpha": 0.7, "boxstyle": "round"})

    # Set axis range and labels
    for axis in axes.ravel():
        if match_scale_x:
            axis.set_xlim(0, furthest_value * 1.1)
        if match_scale_y:
            axis.set_ylim(0, largest_bar * 1.1)
        if x_range is not None:
            if len(x_range) != 2:
                raise ValueError("x_range must be [x_min, x_max].")
            axis.set_xlim(x_range[0], x_range[1])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

    # Save file is a filename has been passed
    if filename is not None:
        save(fig, filename, tight_layout=True)
    else:
        return fig, axes


def plot_state_lifetime_dist(
    stc,
    gamma_shape,
    gamma_scale,
    colors=None,
    filename=None,
):
    """Plots a lifetime distribution for each state.

    This function is a wrapper for `_plot_state_lifetimes` that adds a theoretical
    gamma distribution to the histogram.

    Parameters
    ----------
    stc : np.ndarray
        State time course to analyze.
    gamma_shape : float
        Shape parameter for the gamma distribution.
    gamma_scale : float
        Scale parameter for the gamma distribution.
    colors : list, optional
        List of colors for the histograms. If :code:`None`, a default
        colormap is used.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename=None`.
    axes : plt.axes
        Matplotlib axis object(s). Only returned if :code:`filename=None`.
    """
    fig, axes = _plot_state_lifetimes(stc, colors=colors, y_label="Occurrence")
    for i, axis in enumerate(axes.ravel()):
        # Set axis lables
        if i == stc.shape[-1] - 1:
            axis.set_xlabel("Lifetime (samples)", fontsize=12)
        axis.set_ylabel("Occurrence", fontsize=12)
        axis.tick_params(axis="both", which="both", labelsize=12)

        # Add theoretical gamma distribution
        tmp_axis = axis.twinx()
        x_lim = axis.get_xlim()[1]
        gamma_pdf = gamma.pdf(
            x=np.linspace(0, x_lim, 1000), a=gamma_shape, scale=gamma_scale
        )
        tmp_axis.plot(
            np.linspace(0, x_lim, 1000),
            gamma_pdf,
            color="k",
            linestyle="--",
            linewidth=2,
        )
        tmp_axis.tick_params(
            axis="y", which="both", length=0, labelleft=False, labelright=False
        )
    plt.tight_layout()
    if filename is not None:
        save(fig, filename)
    else:
        return fig, axes


def plot_violin(
    df,
    x_var,
    y_var,
    hue_var,
    palette,
    ylbl=True,
    ylim=None,
    figsize=None,
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
    ylbl : bool, optional
        If :code:`True`, the y-axis label is shown, and the y-ticks are hidden.
        If :code:`False`, the y-axis label is hidden, and the y-ticks are shown.
    ylim : list, optional
        List of two floats specifying the y-axis limits.
    figsize : tuple, optional
        Tuple specifying the figure size (width, height).
        Defaults to (5, 1).
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
        inner="quart",
        density_norm="count",
        legend=False,
        linewidth=1,
        ax=ax,
    )
    sns.despine(ax=ax, top=True, right=True)

    # Adjust tick settings
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # used for numbers < 0.1 or >= 10
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

    # Adjust axis settings
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylbl:
        ax.set_ylabel("")
    else:
        ax.set_yticks([])
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)
    ax.set_xlabel(x_var, fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    ax.tick_params(labelsize=9)

    if filename is not None:
        save(fig, filename)
    else:
        return fig, ax
