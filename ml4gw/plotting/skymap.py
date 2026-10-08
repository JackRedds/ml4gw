"""
Mollweide plots of HEALPix sky localizations, e.g. from the
FITS sky maps written by AMPLFI or HyperWave. Only uses
matplotlib's built-in ``"mollweide"`` projection, so neither
``ligo.skymap`` nor ``healpy`` are required.

Plots are drawn onto a caller-provided axis when one is given,
so that styling is left to the caller and several sky maps
can be overlaid, e.g. to compare two pipelines::

    fig = plt.figure()
    ax = fig.add_subplot(projection="mollweide")
    plot_skymap("amplfi.fits", ax=ax, label="AMPLFI")
    plot_skymap(
        "hyperwave.fits", ax=ax, shade=False,
        contour_color="tab:orange", label="HyperWave",
    )
    ax.legend()
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..skymap import SkyMap


def _wrap(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)"""
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def _ra_to_x(ra, center_ra: float, flip_ra: bool):
    """Map right ascension to the mollweide longitude axis"""
    return _wrap(center_ra - ra) if flip_ra else _wrap(ra - center_ra)


def _x_to_ra(x, center_ra: float, flip_ra: bool):
    return np.mod(center_ra - x if flip_ra else center_ra + x, 2 * np.pi)


def plot_skymap(
    skymap: SkyMap | str | Path,
    ax: Axes | None = None,
    contours: Sequence[float] = (0.5, 0.9),
    true_ra: float | None = None,
    true_dec: float | None = None,
    shade: bool = True,
    cmap: str = "Blues",
    colorbar: bool = True,
    contour_color: str = "black",
    contour_labels: bool = False,
    label: str | None = None,
    center_ra: float = np.pi,
    flip_ra: bool = True,
    resolution: tuple[int, int] = (720, 360),
) -> Axes:
    """
    Plot a sky localization on a Mollweide projection.

    Args:
        skymap:
            The sky map to plot, or the path to
            a FITS file to read it from
        ax:
            Axis with ``projection="mollweide"`` to plot on.
            If ``None``, a new figure and axis are created.
        contours:
            Credible levels at which to draw
            contours. Pass ``()`` to draw none.
        true_ra:
            Right ascension in radians of the true source
            location, which is marked with a star if given
        true_dec:
            Declination in radians of the true source location
        shade:
            Whether to shade the probability density.
            Set to ``False`` when overlaying contours
            on a sky map that has already been plotted.
        cmap: Colormap for the probability density
        colorbar: Whether to draw a colorbar for the shading
        contour_color: Color of the credible region contours
        contour_labels: Whether to label contours with their level
        label:
            Legend label for the contours. Pass this when
            overlaying several sky maps on one axis, then
            call ``ax.legend()``.
        center_ra:
            Right ascension in radians at the center of the
            plot. Defaults to 12h, matching ``ligo.skymap``.
        flip_ra:
            If ``True``, right ascension increases to the
            left (East-left), the usual astronomical convention
        resolution:
            Number of ``(longitude, latitude)`` grid points
            the sky map is evaluated at for plotting

    Returns:
        The axis the sky map was drawn on
    """
    if not isinstance(skymap, SkyMap):
        skymap = SkyMap.from_fits(skymap)

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection="mollweide")
    elif ax.name != "mollweide":
        raise ValueError(
            f"ax must use the 'mollweide' projection, not {ax.name!r}"
        )

    # evaluate the sky map on a regular grid in the plot's coordinates
    nx, ny = resolution
    x_edges = np.linspace(-np.pi, np.pi, nx + 1)
    y_edges = np.linspace(-np.pi / 2, np.pi / 2, ny + 1)
    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x, y)
    idx = skymap.pixel_index(_x_to_ra(xx, center_ra, flip_ra), yy)

    if shade:
        per_deg2 = skymap.probdensity[idx] * (np.pi / 180) ** 2
        mesh = ax.pcolormesh(
            x_edges, y_edges, per_deg2, cmap=cmap, rasterized=True
        )
        if colorbar:
            ax.figure.colorbar(
                mesh,
                ax=ax,
                orientation="horizontal",
                pad=0.08,
                shrink=0.6,
                label=r"Probability per deg$^2$",
            )

    levels = sorted(contours)
    if levels:
        cls = skymap.credible_levels()[idx]
        cs = ax.contour(
            x, y, cls, levels=levels, colors=contour_color, linewidths=1
        )
        if contour_labels:
            ax.clabel(cs, fmt=lambda level: f"{100 * level:g}%", fontsize=8)
        if label is not None:
            # contour sets don't show up in legends, so add an
            # invisible proxy. It needs a (NaN) point, since
            # empty lines can't be drawn on mollweide axes
            ax.plot([np.nan], [np.nan], color=contour_color, lw=1, label=label)

    if true_ra is not None and true_dec is not None:
        ax.plot(
            _ra_to_x(true_ra, center_ra, flip_ra),
            true_dec,
            marker="*",
            markersize=14,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1,
            linestyle="none",
            zorder=5,
        )

    # label longitude ticks with right ascension in hours
    ticks = np.radians(np.arange(-150, 180, 30))
    ax.set_xticks(ticks)
    hours = np.round(np.degrees(_x_to_ra(ticks, center_ra, flip_ra)) / 15)
    ax.set_xticklabels([f"{h % 24:.0f}h" for h in hours])
    ax.set_xlabel("Right ascension")
    ax.set_ylabel("Declination")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    return ax
