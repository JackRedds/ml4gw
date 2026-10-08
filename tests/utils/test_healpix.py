import numpy as np
import pytest

from ml4gw.utils import healpix


@pytest.fixture(params=[1, 2, 8, 64])
def nside(request):
    return request.param


@pytest.fixture(params=[True, False])
def nest(request):
    return request.param


@pytest.fixture
def angles():
    n = 10000
    theta = np.arccos(np.random.uniform(-1, 1, n))
    phi = np.random.uniform(0, 2 * np.pi, n)
    # include the poles and the polar cap / equatorial boundaries
    theta = np.concatenate([theta, [0, np.pi, np.arccos(2 / 3)]])
    phi = np.concatenate([phi, [0, 0, 1]])
    return theta, phi


def test_npix_nside():
    assert healpix.nside2npix(4) == 192
    assert healpix.npix2nside(192) == 4
    assert np.isclose(healpix.nside2pixarea(4) * 192, 4 * np.pi)
    with pytest.raises(ValueError):
        healpix.nside2npix(3)
    with pytest.raises(ValueError):
        healpix.npix2nside(100)


def test_pixel_centers_round_trip(nside, nest):
    ipix = np.arange(healpix.nside2npix(nside))
    theta, phi = healpix.pix2ang(nside, ipix, nest=nest)
    assert (theta >= 0).all() and (theta <= np.pi).all()
    assert (healpix.ang2pix(nside, theta, phi, nest=nest) == ipix).all()


def test_ang2pix_range(nside, nest, angles):
    ipix = healpix.ang2pix(nside, *angles, nest=nest)
    assert ipix.min() >= 0
    assert ipix.max() < healpix.nside2npix(nside)


def test_ring2nest_map(nside):
    ring = np.arange(healpix.nside2npix(nside))
    nested = healpix.ring2nest_map(ring)
    theta, phi = healpix.pix2ang(nside, np.arange(len(ring)), nest=True)
    assert (nested == healpix.ang2pix(nside, theta, phi, nest=False)).all()


def test_uniq_round_trip():
    order = np.random.randint(0, 30, size=1000)
    ipix = (np.random.rand(1000) * 12 * 4.0**order).astype(np.int64)
    uniq = healpix.nest2uniq(order, ipix)
    assert (uniq == 4 * 4 ** order.astype(object) + ipix).all()
    order_, ipix_ = healpix.uniq2nest(uniq)
    assert (order_ == order).all()
    assert (ipix_ == ipix).all()


@pytest.mark.parametrize("nside", [1, 64, 2**20])
def test_against_healpy(nside, nest, angles):
    hp = pytest.importorskip("healpy")
    theta, phi = angles
    expected = hp.ang2pix(nside, theta, phi, nest=nest)
    assert (healpix.ang2pix(nside, theta, phi, nest=nest) == expected).all()

    ipix = np.random.randint(0, healpix.nside2npix(nside), size=1000)
    theta, phi = healpix.pix2ang(nside, ipix, nest=nest)
    theta_, phi_ = hp.pix2ang(nside, ipix, nest=nest)
    assert np.allclose(theta, theta_)
    assert np.allclose(np.mod(phi, 2 * np.pi), np.mod(phi_, 2 * np.pi))
