import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml4gw.plotting import plot_skymap  # noqa: E402
from ml4gw.skymap import SkyMap  # noqa: E402
from ml4gw.utils import healpix  # noqa: E402


@pytest.fixture
def skymap():
    nside = 16
    theta, phi = healpix.pix2ang(nside, np.arange(healpix.nside2npix(nside)))
    prob = np.exp(-((theta - 1.2) ** 2 + (phi - 2.0) ** 2) / 0.05)
    return SkyMap.from_healpix(prob / prob.sum())


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_skymap(skymap):
    ax = plot_skymap(
        skymap, true_ra=2.0, true_dec=np.pi / 2 - 1.2, resolution=(180, 90)
    )
    assert ax.name == "mollweide"
    assert len(ax.collections) > 1  # density mesh + contours
    labels = [t.get_text() for t in ax.get_xticklabels()]
    # centered on 12h with right ascension increasing to the left
    assert labels[len(labels) // 2] == "12h"
    assert labels[0] == "22h"
    assert len(ax.figure.axes) == 2  # plot + colorbar


def test_plot_skymap_overlay(skymap, tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "skymap.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column("UNIQ", "K", array=skymap.uniq),
            fits.Column("PROBDENSITY", "D", array=skymap.probdensity),
        ],
        header=fits.Header({"ORDERING": "NUNIQ"}),
    ).writeto(path)

    fig = plt.figure()
    ax = fig.add_subplot(projection="mollweide")
    plot_skymap(skymap, ax=ax, label="a", resolution=(180, 90))
    plot_skymap(
        path,
        ax=ax,
        shade=False,
        contour_color="tab:orange",
        label="b",
        resolution=(180, 90),
    )
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["a", "b"]
    fig.savefig(tmp_path / "skymap.png")


def test_plot_skymap_wrong_projection(skymap):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="mollweide"):
        plot_skymap(skymap, ax=ax)
