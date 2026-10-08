"""
Minimal numpy implementations of the HEALPix pixel-indexing
routines needed to read and plot sky maps, so that doing so
doesn't require ``healpy`` or ``ligo.skymap``.

Algorithms follow Górski et al. (2005) and the reference
HEALPix C++ implementation (``healpix_base.cc``).
All angles are in radians, with ``theta`` the colatitude
in ``[0, pi]`` and ``phi`` the longitude (right ascension).
"""

import numpy as np

# row and column offsets of the 12 base faces
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def _check_nside(nside: int) -> int:
    nside = int(nside)
    if nside < 1 or nside & (nside - 1):
        raise ValueError(f"nside must be a power of 2, got {nside}")
    return nside


def nside2npix(nside: int) -> int:
    """Number of pixels in a HEALPix map with the given ``nside``"""
    return 12 * _check_nside(nside) ** 2


def npix2nside(npix: int) -> int:
    """HEALPix ``nside`` of a map with ``npix`` pixels"""
    nside = int(round(np.sqrt(npix / 12)))
    if 12 * nside**2 != npix:
        raise ValueError(f"{npix} is not a valid number of HEALPix pixels")
    return _check_nside(nside)


def nside2pixarea(nside: int) -> float:
    """Solid angle of a single pixel in steradians"""
    return 4 * np.pi / nside2npix(nside)


def uniq2nest(uniq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert multi-order ``UNIQ`` pixel indices to
    HEALPix orders and NESTED pixel indices.

    Args:
        uniq: ``UNIQ`` pixel indices, i.e. ``4 * nside**2 + ipix``

    Returns:
        Tuple of the order (``log2(nside)``) and NESTED
        pixel index of each pixel
    """
    uniq = np.asarray(uniq, dtype=np.int64)
    if np.any(uniq < 4):
        raise ValueError("UNIQ pixel indices must be >= 4")
    # order is half the index of the most significant bit, minus
    # one. log2 can round up just below a power of 4 for large
    # orders, so correct for that with exact integer comparisons
    order = (np.log2(uniq).astype(np.int64) >> 1) - 1
    order -= (np.int64(4) << (2 * order)) > uniq
    order += (np.int64(4) << (2 * order + 2)) <= uniq
    ipix = uniq - (np.int64(4) << (2 * order))
    return order, ipix


def nest2uniq(order: np.ndarray, ipix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`uniq2nest`"""
    order = np.asarray(order, dtype=np.int64)
    ipix = np.asarray(ipix, dtype=np.int64)
    return (np.int64(4) << (2 * order)) + ipix


def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Interleave zeros between the bits of ``v`` (Morton encoding)"""
    v = v.astype(np.uint64)
    out = np.zeros_like(v)
    for i in range(32):
        out |= ((v >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i)
    return out


def _compress_bits(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_spread_bits`, keeping the even bits"""
    v = v.astype(np.uint64)
    out = np.zeros_like(v)
    for i in range(32):
        out |= ((v >> np.uint64(2 * i)) & np.uint64(1)) << np.uint64(i)
    return out


def _ang2xyf(nside: int, theta: np.ndarray, phi: np.ndarray):
    """Map angles to (x, y, face) coordinates of the NESTED scheme"""
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2 * np.pi) * (2 / np.pi)  # in [0, 4)

    # equatorial region
    temp1 = nside * (0.5 + tt)
    temp2 = nside * z * 0.75
    jp = np.floor(temp1 - temp2).astype(np.int64)
    jm = np.floor(temp1 + temp2).astype(np.int64)
    ifp = jp // nside
    ifm = jm // nside
    face_eq = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix_eq = jm & (nside - 1)
    iy_eq = nside - (jp & (nside - 1)) - 1

    # polar caps
    ntt = np.minimum(np.floor(tt).astype(np.int64), 3)
    tp = tt - ntt
    tmp = nside * np.sqrt(3 * (1 - za))
    jp = np.minimum(np.floor(tp * tmp).astype(np.int64), nside - 1)
    jm = np.minimum(np.floor((1 - tp) * tmp).astype(np.int64), nside - 1)
    north = z > 0
    face_pol = np.where(north, ntt, ntt + 8)
    ix_pol = np.where(north, nside - jm - 1, jp)
    iy_pol = np.where(north, nside - jp - 1, jm)

    equatorial = za <= 2 / 3
    face = np.where(equatorial, face_eq, face_pol)
    ix = np.where(equatorial, ix_eq, ix_pol)
    iy = np.where(equatorial, iy_eq, iy_pol)
    return ix, iy, face


def ang2pix(
    nside: int, theta: np.ndarray, phi: np.ndarray, nest: bool = True
) -> np.ndarray:
    """
    HEALPix pixel index containing each of the given angles.

    Args:
        nside: HEALPix resolution parameter
        theta: Colatitude in radians, in ``[0, pi]``
        phi: Longitude in radians
        nest:
            If ``True``, return NESTED indices,
            otherwise return RING indices

    Returns:
        Array of pixel indices with the broadcast
        shape of ``theta`` and ``phi``
    """
    nside = _check_nside(nside)
    theta, phi = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
    )
    if nest:
        ix, iy, face = _ang2xyf(nside, theta, phi)
        sub = _spread_bits(ix) | (_spread_bits(iy) << np.uint64(1))
        return face * np.int64(nside) ** 2 + sub.astype(np.int64)
    return _ang2pix_ring(nside, theta, phi)


def _ang2pix_ring(nside: int, theta: np.ndarray, phi: np.ndarray):
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2 * np.pi) * (2 / np.pi)
    nl4 = 4 * nside
    npix = 12 * nside**2
    ncap = 2 * nside * (nside - 1)

    # equatorial region
    temp1 = nside * (0.5 + tt)
    temp2 = nside * z * 0.75
    jp = np.floor(temp1 - temp2).astype(np.int64)
    jm = np.floor(temp1 + temp2).astype(np.int64)
    ir = nside + 1 + jp - jm
    kshift = 1 - (ir & 1)
    ip = np.mod((jp + jm - nside + kshift + 1) // 2, nl4)
    pix_eq = ncap + (ir - 1) * nl4 + ip

    # polar caps
    tp = tt - np.floor(tt)
    tmp = nside * np.sqrt(3 * (1 - za))
    jp = np.floor(tp * tmp).astype(np.int64)
    jm = np.floor((1 - tp) * tmp).astype(np.int64)
    ir = jp + jm + 1
    ip = np.mod(np.floor(tt * ir).astype(np.int64), 4 * ir)
    pix_pol = np.where(
        z > 0, 2 * ir * (ir - 1) + ip, npix - 2 * ir * (ir + 1) + ip
    )
    return np.where(za <= 2 / 3, pix_eq, pix_pol)


def pix2ang(
    nside: int, ipix: np.ndarray, nest: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Angular coordinates of the centers of the given pixels.

    Args:
        nside: HEALPix resolution parameter
        ipix: Pixel indices
        nest:
            If ``True``, ``ipix`` are NESTED indices,
            otherwise they are RING indices

    Returns:
        Tuple of the colatitude ``theta`` and
        longitude ``phi`` of each pixel center, in radians
    """
    nside = _check_nside(nside)
    ipix = np.asarray(ipix, dtype=np.int64)
    if not nest:
        return _pix2ang_ring(nside, ipix)

    npface = nside**2
    face = ipix // npface
    sub = (ipix & (npface - 1)).astype(np.uint64)
    ix = _compress_bits(sub).astype(np.int64)
    iy = _compress_bits(sub >> np.uint64(1)).astype(np.int64)

    # ring number, counted from the north pole
    jr = _JRLL[face] * nside - ix - iy - 1
    fact2 = 4 / (12 * npface)
    nr = np.where(
        jr < nside, jr, np.where(jr > 3 * nside, 4 * nside - jr, nside)
    )
    z = np.where(
        jr < nside,
        1 - nr**2 * fact2,
        np.where(
            jr > 3 * nside,
            nr**2 * fact2 - 1,
            (2 * nside - jr) * 2 * nside * fact2,
        ),
    )
    kshift = np.where(nr == nside, (jr - nside) & 1, 0)
    jp = (_JPLL[face] * nr + ix - iy + 1 + kshift) // 2
    jp = np.where(jp > 4 * nside, jp - 4 * nside, jp)
    jp = np.where(jp < 1, jp + 4 * nside, jp)
    phi = (jp - (kshift + 1) * 0.5) * (np.pi / 2 / nr)
    return np.arccos(np.clip(z, -1, 1)), phi


def _pix2ang_ring(nside: int, ipix: np.ndarray):
    npix = 12 * nside**2
    ncap = 2 * nside * (nside - 1)
    fact2 = 4 / npix

    # north polar cap
    iring_n = (1 + np.floor(np.sqrt(1 + 2 * ipix)).astype(np.int64)) >> 1
    iphi_n = ipix + 1 - 2 * iring_n * (iring_n - 1)
    z_n = 1 - iring_n**2 * fact2
    phi_n = (iphi_n - 0.5) * (np.pi / 2 / iring_n)

    # equatorial region
    ip = ipix - ncap
    iring_e = ip // (4 * nside) + nside
    iphi_e = ip % (4 * nside) + 1
    fodd = np.where((iring_e + nside) & 1, 1.0, 0.5)
    z_e = (2 * nside - iring_e) * 2 * nside * fact2
    phi_e = (iphi_e - fodd) * (np.pi / 2 / nside)

    # south polar cap
    ip = npix - ipix
    iring_s = (1 + np.floor(np.sqrt(2 * ip - 1)).astype(np.int64)) >> 1
    iphi_s = 4 * iring_s + 1 - (ip - 2 * iring_s * (iring_s - 1))
    z_s = -1 + iring_s**2 * fact2
    phi_s = (iphi_s - 0.5) * (np.pi / 2 / iring_s)

    north = ipix < ncap
    south = ipix >= npix - ncap
    z = np.where(north, z_n, np.where(south, z_s, z_e))
    phi = np.where(north, phi_n, np.where(south, phi_s, phi_e))
    return np.arccos(np.clip(z, -1, 1)), phi


def ring2nest_map(m: np.ndarray) -> np.ndarray:
    """Reorder a full-sky RING-ordered map into NESTED ordering"""
    m = np.asarray(m)
    nside = npix2nside(len(m))
    theta, phi = pix2ang(nside, np.arange(len(m)), nest=True)
    return m[ang2pix(nside, theta, phi, nest=False)]
