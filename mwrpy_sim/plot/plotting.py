import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure

from .plot_meta import ATTRIBUTES, PlotMeta


def plot_sim_data(
    sim_data: netCDF4.Dataset,
    site: str,
    source: str,
    date_d: str | None,
    start_d: str,
    stop_d: str,
    save_path: str,
    dpi: int = 120,
    show: bool = False,
):
    """Plots simulation data from a netCDF4 dataset.

    :param sim_data: Dataset containing simulation data.
    :param site: Name of site.
    :param source: Name of the data source.
    :param dpi: Dots per inch for the plot resolution.
    :param show: If True, display the plot interactively.
    :return:
    """
    fig, axes = _initialize_figure(3, 3, dpi)
    for ax, (var_name, meta) in zip(axes, ATTRIBUTES.items()):
        if var_name not in sim_data.variables and meta.source == "profile":
            data = np.ones((1, sim_data.dimensions["height"].size)) * np.nan
        elif var_name not in sim_data.variables and meta.source == "1d":
            data = np.ones((1,)) * np.nan
        else:
            data = sim_data.variables[var_name][:].data
            data = np.ma.masked_equal(data, -999.0)
            data *= 1000.0 if "lwc" in var_name else 1.0
        if data.ndim == 1:
            _plot_histogram(ax, data, meta)
        elif data.ndim == 2:
            _plot_profile(ax, data, sim_data.variables["height"][:].data)

        _set_axis(ax, meta)

    # Set the title for the figure
    _set_title(axes, source, site, date_d, start_d, stop_d)

    # Save the figure
    _handle_saving(site, source, date_d, start_d, stop_d, save_path, show)
    plt.close(fig)


def _initialize_figure(ncols: int, nrows: int, dpi: int) -> tuple[Figure, list[Axes]]:
    """Creates an empty figure according to the number of subplots."""
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(12, 12),
        dpi=dpi,
        facecolor="white",
    )
    fig.subplots_adjust(left=0.06, right=0.73)
    return fig, axes.flatten()


def _handle_saving(
    site: str,
    source: str,
    date_d: str | None,
    start_d: str,
    stop_d: str,
    save_path,
    show: bool,
) -> None:
    """Saves plot."""
    if source == "standard_atmosphere":
        filename = f"mwrpy_sim_{site}.png"
    else:
        filename = (
            f"mwrpy_sim_{site}_{source}_{date_d}.png"
            if date_d
            else f"mwrpy_sim_{site}_{source}_{start_d}_{stop_d}.png"
        )
    plt.savefig(
        str(os.path.join(save_path + filename)),
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()


def _set_axis(ax, meta: PlotMeta) -> None:
    """Sets axis range and labels defined in PlotMeta."""
    pos = ax.get_position()
    y_label = "Height (m)" if meta.source == "profile" else "Log frequency"
    if pos.x0 < 0.1:
        ax.set_ylabel(y_label)
    elif pos.x1 > 0.1 and meta.source == "profile":
        ax.set_yticklabels([])
    ax.set_xlabel(f"{meta.name} ({meta.xlabel})")
    ax.set_xlim(meta.plot_range)
    if meta.source == "profile":
        ax.set_ylim(0, 10000)


def _set_title(
    axes, source: str, site: str, date_d: str | None, start_d: str, stop_d: str
) -> None:
    """Set the title for the figure."""
    if source == "standard_atmosphere":
        axes[1].set_title(
            f"MWRpy sim data (source: {source})",
            fontsize=16,
        )
    else:
        axes[1].set_title(
            f"MWRpy sim data for {site} (source: {source})\nfrom {start_d} to {stop_d}"
            if not date_d
            else f"MWRpy simulation data for {site} ({source})\non {date_d}",
            fontsize=16,
        )


def _plot_histogram(ax, data: np.ndarray, meta: PlotMeta) -> None:
    """Plots a histogram of the data."""
    assert meta.plot_range is not None, "Plot range must be defined in PlotMeta."
    ax.hist(
        data,
        bins=np.linspace(meta.plot_range[0], meta.plot_range[1], 25),
        density=True,
        alpha=0.6,
        color="black",
        linewidth=1.0,
        histtype="step",
    )
    ax.set_yscale("log")
    ax.text(
        0.05,
        0.1,
        f"mean: {np.ma.mean(data):.2f} {meta.xlabel}\n",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )


def _plot_profile(ax, data: np.ndarray, height: np.ndarray) -> None:
    """Plots a mean profile and stdev of the data."""
    ax.fill_betweenx(
        height,
        np.ma.mean(data, axis=0) - np.ma.std(data, axis=0),
        np.ma.mean(data, axis=0) + np.ma.std(data, axis=0),
        color="black",
        alpha=0.3,
    )
    ax.plot(
        np.ma.mean(data, axis=0),
        height,
        color="black",
        linewidth=1.0,
    )
