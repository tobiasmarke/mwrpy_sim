"""Metadata module for plot."""

from typing import NamedTuple


class PlotMeta(NamedTuple):
    """Class for plot module."""

    name: str | None = None
    xlabel: str | None = None
    plot_range: tuple[float, float] | None = None
    source: str | None = None


_K = "K"
_KGM2 = "kg m$^{-2}$"
_KGM3 = "kg m$^{-3}$"
_GM2 = "g m$^{-2}$"
_GM3 = "g m$^{-3}$"
_1 = "1"
_Pa = "Pa"

# Attributes define order of plots.
ATTRIBUTES = {
    "lwp": PlotMeta(
        name="LWP",
        xlabel=_KGM2,
        plot_range=(0.0, 1.8),
        source="1d",
    ),
    "lwp_pro": PlotMeta(
        name="LWP (prognostic)",
        xlabel=_KGM2,
        plot_range=(0.0, 1.8),
        source="1d",
    ),
    "iwv": PlotMeta(
        name="IWV",
        xlabel=_KGM2,
        plot_range=(0, 50),
        source="1d",
    ),
    "absolute_humidity": PlotMeta(
        name="Absolute humidity",
        xlabel=_KGM3,
        plot_range=(0.0, 0.015),
        source="profile",
    ),
    "air_temperature": PlotMeta(
        name="Air temperature",
        xlabel=_K,
        plot_range=(200.0, 320.0),
        source="profile",
    ),
    "relative_humidity": PlotMeta(
        name="Relative humidity",
        xlabel=_1,
        plot_range=(0.0, 1.3),
        source="profile",
    ),
    "air_pressure": PlotMeta(
        name="Air pressure",
        xlabel=_Pa,
        plot_range=(0.0, 120000.0),
        source="profile",
    ),
    "lwc": PlotMeta(
        name="LWC",
        xlabel=_GM3,
        plot_range=(0.0, 0.5),
        source="profile",
    ),
    "lwc_pro": PlotMeta(
        name="LWC (prognostic)",
        xlabel=_GM3,
        plot_range=(0.0, 0.5),
        source="profile",
    ),
}
