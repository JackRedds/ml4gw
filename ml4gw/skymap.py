"""
A lightweight container for HEALPix sky localizations,
with a reader for the FITS sky maps written by
``ligo.skymap`` (and so by AMPLFI), ``healpy``, or any
other package following the LIGO/Virgo/KAGRA sky map format.

Both multi-order (``ORDERING = NUNIQ``) and flat
(``ORDERING = NESTED`` or ``RING``) maps are supported.
Only ``numpy`` and ``astropy`` are required; neither
``healpy`` nor ``ligo.skymap`` are needed.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import healpix

# column names used for the per-pixel probability in flat maps,
# in order of preference. healpy writes unnamed columns as "T"
_FLAT_PROB_COLUMNS = ("PROB", "PROBABILITY", "PROBDENSITY", "T")


@dataclass
class SkyMap:
    """
    A sky localization stored as a multi-order HEALPix map.

    Args:
        uniq:
            ``UNIQ`` indices of each pixel. The pixels
            should tile the whole sky without overlapping.
        probdensity:
            Probability per steradian in each pixel
        meta:
            Header metadata, e.g. read from a FITS file
    """

    uniq: np.ndarray
    probdensity: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.uniq = np.asarray(self.uniq, dtype=np.int64)
        self.probdensity = np.asarray(self.probdensity, dtype=np.float64)
        if self.uniq.shape != self.probdensity.shape:
            raise ValueError(
                "uniq and probdensity must have the same shape, got "
                f"{self.uniq.shape} and {self.probdensity.shape}"
            )
        self._order, self._ipix = healpix.uniq2nest(self.uniq)
        self._max_order = int(self._order.max())

        if len(self.uniq) == 12 * 4**self._max_order:
            # a flat map, where pixels can be looked up directly by
            # their NESTED index. Skips the search structure below,
            # which matters for high resolution maps
            self._sorted_start = None
            ascending = (self._ipix[1:] > self._ipix[:-1]).all()
            self._sort_idx = None if ascending else np.argsort(self._ipix)
        else:
            # precompute the ranges each pixel covers at the finest
            # order, so that arbitrary sky locations can be looked
            # up with a binary search
            start = self._ipix << (2 * (self._max_order - self._order))
            self._sort_idx = np.argsort(start)
            self._sorted_start = start[self._sort_idx]

    @classmethod
    def from_healpix(
        cls,
        prob: np.ndarray,
        nest: bool = True,
        meta: dict | None = None,
    ) -> "SkyMap":
        """
        Build a ``SkyMap`` from a flat, full-sky HEALPix map.

        Args:
            prob: Probability contained in each pixel
            nest:
                Whether ``prob`` is in NESTED (``True``)
                or RING (``False``) ordering
            meta: Optional metadata to attach
        """
        prob = np.asarray(prob, dtype=np.float64).ravel()
        nside = healpix.npix2nside(len(prob))
        if not nest:
            prob = healpix.ring2nest_map(prob)
        order = int(np.log2(nside))
        uniq = healpix.nest2uniq(order, np.arange(len(prob)))
        probdensity = prob / healpix.nside2pixarea(nside)
        return cls(uniq, probdensity, dict(meta or {}))

    @classmethod
    def from_fits(cls, path: str | Path, hdu: int = 1) -> "SkyMap":
        """
        Read a sky map from a (possibly gzipped) FITS file.

        Args:
            path: Path to the FITS file
            hdu: Index of the HDU containing the HEALPix table
        """
        from astropy.io import fits

        with fits.open(path) as hdul:
            header = hdul[hdu].header
            data = hdul[hdu].data
            meta = dict(header)
            ordering = str(header.get("ORDERING", "")).upper()
            names = [name.upper() for name in data.columns.names]

            if ordering == "NUNIQ" or "UNIQ" in names:
                # copy the columns we need, so that the (possibly
                # large) table can be freed once the file is closed
                return cls(
                    np.array(data["UNIQ"], dtype=np.int64),
                    np.array(data["PROBDENSITY"], dtype=np.float64),
                    meta,
                )

            if str(header.get("INDXSCHM", "IMPLICIT")).upper() != "IMPLICIT":
                raise ValueError(
                    "Partial-sky (explicitly indexed) flat "
                    f"HEALPix maps are not supported: {path}"
                )
            if ordering not in ("NESTED", "RING"):
                raise ValueError(
                    f"Unrecognized HEALPix ORDERING {ordering!r} in {path}"
                )

            column = next(
                (c for c in _FLAT_PROB_COLUMNS if c in names), names[0]
            )
            values = np.array(data[column], dtype=np.float64).ravel()
            if column == "PROBDENSITY":
                nside = healpix.npix2nside(len(values))
                values = values * healpix.nside2pixarea(nside)
        return cls.from_healpix(values, nest=ordering == "NESTED", meta=meta)

    @property
    def order(self) -> np.ndarray:
        """HEALPix order (``log2(nside)``) of each pixel"""
        return self._order

    @property
    def ipix(self) -> np.ndarray:
        """NESTED index of each pixel at its own order"""
        return self._ipix

    @property
    def pixel_area(self) -> np.ndarray:
        """Solid angle of each pixel in steradians"""
        return np.pi / 3 / 4.0**self._order

    @property
    def prob(self) -> np.ndarray:
        """Probability contained in each pixel"""
        return self.probdensity * self.pixel_area

    def pixel_index(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """
        Index of the pixel containing each sky location.

        Args:
            ra: Right ascension in radians
            dec: Declination in radians
        """
        ra, dec = np.broadcast_arrays(np.asarray(ra), np.asarray(dec))
        idx = healpix.ang2pix(
            1 << self._max_order, np.pi / 2 - dec, ra, nest=True
        )
        if self._sorted_start is not None:
            idx = np.searchsorted(self._sorted_start, idx, side="right") - 1
        return idx if self._sort_idx is None else self._sort_idx[idx]

    def density(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """
        Probability per steradian at each sky location.

        Args:
            ra: Right ascension in radians
            dec: Declination in radians
        """
        return self.probdensity[self.pixel_index(ra, dec)]

    def credible_levels(self) -> np.ndarray:
        """
        Credible level of each pixel: the total probability
        contained in all pixels at least as dense as it,
        i.e. the smallest credible region containing it.
        """
        rank = np.argsort(self.probdensity, kind="stable")[::-1]
        prob = self.prob
        cls = np.empty_like(prob)
        cls[rank] = np.cumsum(prob[rank]) / prob.sum()
        return cls

    def credible_level(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """
        Credible level at each sky location, also called the
        searched probability for the location of a known source.

        Args:
            ra: Right ascension in radians
            dec: Declination in radians
        """
        return self.credible_levels()[self.pixel_index(ra, dec)]

    def credible_area(self, level: float) -> float:
        """
        Area in square degrees of the smallest region
        containing a fraction ``level`` of the probability.
        """
        rank = np.argsort(self.probdensity, kind="stable")[::-1]
        prob = self.prob[rank]
        cumulative = np.cumsum(prob) / prob.sum()
        n = np.searchsorted(cumulative, level) + 1
        return self.pixel_area[rank][:n].sum() * (180 / np.pi) ** 2

    def searched_area(self, ra: float, dec: float) -> float:
        """
        Area in square degrees of the smallest credible
        region containing the given sky location.
        """
        dens = self.density(ra, dec)
        mask = self.probdensity >= dens
        return self.pixel_area[mask].sum() * (180 / np.pi) ** 2
