import numpy as np
import pytest

from ml4gw.skymap import SkyMap
from ml4gw.utils import healpix

fits = pytest.importorskip("astropy.io.fits")

NSIDE = 32
SIGMA = 0.1
FULL_SKY = 4 * np.pi * (180 / np.pi) ** 2


@pytest.fixture
def source():
    return 1.0, 0.3


@pytest.fixture
def flat_prob(source):
    """Gaussian blob on the sky around ``source``, NESTED"""
    ra, dec = source
    npix = healpix.nside2npix(NSIDE)
    theta, phi = healpix.pix2ang(NSIDE, np.arange(npix))
    cos_dist = np.sin(dec) * np.cos(theta) + np.cos(dec) * np.sin(
        theta
    ) * np.cos(phi - ra)
    prob = np.exp((cos_dist - 1) / SIGMA**2)
    return prob / prob.sum()


def coarsen(skymap, fraction=0.5):
    """
    Merge the groups of 4 sibling pixels containing the least
    probability into their parents, to build a multi-order map
    """
    order, ipix = skymap.order, skymap.ipix
    parent = ipix >> 2
    parent_prob = np.bincount(parent, weights=skymap.prob)
    merged = np.argsort(parent_prob)[: int(fraction * len(parent_prob))]
    keep = ~np.isin(parent, merged)
    uniq = np.concatenate(
        [
            skymap.uniq[keep],
            healpix.nest2uniq(order[0] - 1, merged),
        ]
    )
    area = healpix.nside2pixarea(1 << (order[0] - 1))
    density = np.concatenate(
        [skymap.probdensity[keep], parent_prob[merged] / area]
    )
    perm = np.random.permutation(len(uniq))
    return SkyMap(uniq[perm], density[perm])


def test_from_healpix(flat_prob, source):
    skymap = SkyMap.from_healpix(flat_prob)
    assert (skymap.order == np.log2(NSIDE)).all()
    assert np.isclose(skymap.prob.sum(), 1)
    assert np.isclose(skymap.pixel_area.sum(), 4 * np.pi)

    peak = skymap.density(*source)
    assert peak == skymap.probdensity.max()
    # only the pixel containing the source is more probable
    assert skymap.credible_level(*source) < 0.05

    # a ring-ordered map of the same probabilities is equivalent
    ring = np.empty_like(flat_prob)
    ring[
        healpix.ang2pix(
            NSIDE, *healpix.pix2ang(NSIDE, np.arange(len(ring))), nest=False
        )
    ] = flat_prob
    ring_skymap = SkyMap.from_healpix(ring, nest=False)
    assert np.allclose(ring_skymap.probdensity, skymap.probdensity)


def test_unsorted_flat(flat_prob):
    flat = SkyMap.from_healpix(flat_prob)
    perm = np.random.permutation(len(flat.uniq))
    shuffled = SkyMap(flat.uniq[perm], flat.probdensity[perm])
    ra = np.random.uniform(0, 2 * np.pi, 1000)
    dec = np.arcsin(np.random.uniform(-1, 1, 1000))
    assert np.allclose(shuffled.density(ra, dec), flat.density(ra, dec))


def test_credible_area(flat_prob):
    skymap = SkyMap.from_healpix(flat_prob)
    areas = [skymap.credible_area(level) for level in (0.5, 0.9, 1.0)]
    assert areas[0] < areas[1] < areas[2] <= FULL_SKY + 1e-6

    # 2D gaussian: the p credible region has radius
    # sigma * sqrt(-2 log(1 - p)), up to pixelization
    expected = np.pi * SIGMA**2 * -2 * np.log(0.1) * (180 / np.pi) ** 2
    assert np.isclose(areas[1], expected, rtol=0.1)

    uniform = SkyMap.from_healpix(np.ones(12))
    assert np.isclose(uniform.credible_area(0.5), FULL_SKY / 2)


def test_searched_area(flat_prob, source):
    skymap = SkyMap.from_healpix(flat_prob)
    level = skymap.credible_level(*source)
    assert skymap.searched_area(*source) <= skymap.credible_area(level) + 1e-6
    # the antipode is in the least probable pixels
    ra, dec = source
    antipode = np.mod(ra + np.pi, 2 * np.pi), -dec
    assert skymap.searched_area(*antipode) > 0.99 * FULL_SKY


def test_multi_order(flat_prob):
    flat = SkyMap.from_healpix(flat_prob)
    moc = coarsen(flat)
    assert len(moc.uniq) < len(flat.uniq)
    assert np.isclose(moc.prob.sum(), 1)
    assert np.isclose(moc.pixel_area.sum(), 4 * np.pi)

    # lookups agree wherever pixels weren't merged
    ra = np.random.uniform(0, 2 * np.pi, 5000)
    dec = np.arcsin(np.random.uniform(-1, 1, 5000))
    fine = moc.order[moc.pixel_index(ra, dec)] == flat.order[0]
    assert fine.any()
    assert np.allclose(moc.density(ra, dec)[fine], flat.density(ra, dec)[fine])
    assert np.isclose(
        moc.credible_area(0.9), flat.credible_area(0.9), rtol=0.01
    )


def write_moc(path, skymap):
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column("UNIQ", "K", array=skymap.uniq),
            fits.Column(
                "PROBDENSITY", "D", unit="sr-1", array=skymap.probdensity
            ),
        ]
    )
    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = "NUNIQ"
    hdu.header["INDXSCHM"] = "EXPLICIT"
    hdu.header["OBJECT"] = "test"
    hdu.writeto(path)


def write_flat(path, prob, ordering, column="PROB"):
    hdu = fits.BinTableHDU.from_columns([fits.Column(column, "D", array=prob)])
    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = ordering
    hdu.header["NSIDE"] = healpix.npix2nside(len(prob))
    hdu.header["INDXSCHM"] = "IMPLICIT"
    hdu.writeto(path)


def test_from_fits_moc(tmp_path, flat_prob):
    expected = coarsen(SkyMap.from_healpix(flat_prob))
    write_moc(tmp_path / "moc.fits", expected)
    skymap = SkyMap.from_fits(tmp_path / "moc.fits")
    assert (skymap.uniq == expected.uniq).all()
    assert np.allclose(skymap.probdensity, expected.probdensity)
    assert skymap.meta["OBJECT"] == "test"


@pytest.mark.parametrize("column", ["PROB", "T"])
def test_from_fits_flat(tmp_path, flat_prob, column):
    expected = SkyMap.from_healpix(flat_prob)
    write_flat(tmp_path / "nest.fits", flat_prob, "NESTED", column)
    skymap = SkyMap.from_fits(tmp_path / "nest.fits")
    assert np.allclose(skymap.probdensity, expected.probdensity)

    theta, phi = healpix.pix2ang(NSIDE, np.arange(len(flat_prob)))
    ring = np.empty_like(flat_prob)
    ring[healpix.ang2pix(NSIDE, theta, phi, nest=False)] = flat_prob
    write_flat(tmp_path / "ring.fits", ring, "RING", column)
    skymap = SkyMap.from_fits(tmp_path / "ring.fits")
    assert np.allclose(skymap.probdensity, expected.probdensity)


def test_from_fits_bad_ordering(tmp_path, flat_prob):
    write_flat(tmp_path / "bad.fits", flat_prob, "SPIRAL")
    with pytest.raises(ValueError, match="ORDERING"):
        SkyMap.from_fits(tmp_path / "bad.fits")
