from collections.abc import Callable
from typing import Optional, Tuple

import torch

from ml4gw.transforms import SnrRescaler, SpectralDensity, Whiten
from ml4gw.transforms import WaveformProjector
from ml4gw.types import BatchTensor, WaveformTensor

Tensor = torch.Tensor


class PsdEstimator(torch.nn.Module):
    """
    Module that takes a sample of data, splits it into
    two unequal-length segments, calculates the PSD of
    the first section, then returns this PSD along with
    the second section.

    Args:
        length:
            The length, in seconds, of timeseries data
            to be returned for whitening. Note that the
            length of time used for the PSD will then be
            whatever remains along first part of the time
            axis of the input.
        sample_rate:
            Rate at which input data has been sampled in Hz
        fftlength:
            Length of FFTs to use when computing the PSD
        overlap:
            Amount of overlap between FFT windows when
            computing the PSD. Default value of `None`
            uses `fftlength / 2`
        average:
            Method for aggregating spectra from FFT
            windows, either `"mean"` or `"median"`
        fast:
            If `True`, use a slightly faster PSD algorithm
            that is inaccurate for the lowest two frequency
            bins. If you plan on highpassing later, this
            should be fine.
    """

    def __init__(
        self,
        length: float,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "mean",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, fast=fast
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        splits = [X.size(-1) - self.size, self.size]
        background, X = torch.split(X, splits, dim=-1)

        # if we have 2 batch elements in our input data,
        # it will be assumed that the 0th element corresponds
        # to true background and the 1th element corresponds
        # to injected data, in which case we'll only compute
        # the background PSD on the former
        if X.ndim == 3 and X.size(0) == 2:
            background = background[0]

        self.spectral_density.to(device=background.device)
        psds = self.spectral_density(background.double())
        return X, psds


class Injector(torch.nn.Module):
    """
    Module that injects gravitational waveforms into a batch of
    background interferometer data.

    An ``Injector`` orchestrates, but does not itself implement,
    the individual steps of an injection: it projects raw
    polarizations onto a network of interferometers with a
    :class:`~ml4gw.transforms.WaveformProjector`, optionally
    rescales the resulting responses to a target network SNR
    with a :class:`~ml4gw.transforms.SnrRescaler`, adds them to
    a background timeseries, then whitens the result with a
    :class:`~ml4gw.transforms.Whiten`. This keeps ``Injector``
    agnostic to how any downstream package chooses to configure,
    fit, or sample from each of these pieces.

    Extrinsic sky parameters (``dec``, ``psi``, ``phi``) can either
    be passed directly at call time, or sampled on the fly by
    providing ``extrinsics_sampler``, a callable that maps a number
    of waveforms ``N`` to a ``(dec, psi, phi)`` tuple of tensors of
    that length. This leaves the choice of sky-location distribution
    entirely up to the caller.

    Args:
        projector:
            :class:`~ml4gw.transforms.WaveformProjector` used to
            project polarizations onto an interferometer network
        whitener:
            :class:`~ml4gw.transforms.Whiten` used to whiten the
            data after injection
        extrinsics_sampler:
            Optional callable mapping a number of waveforms ``N``
            to a tuple of ``(dec, psi, phi)`` tensors of length
            ``N``. Only used if ``dec``, ``psi``, and ``phi`` are
            not passed directly at call time.
        snr_rescaler:
            Optional, already-fit :class:`~ml4gw.transforms.SnrRescaler`
            used to rescale projected waveforms to a target network
            SNR before injection. If left as ``None``, waveforms are
            injected at their native amplitude and no SNRs are
            returned.
    """

    def __init__(
        self,
        projector: WaveformProjector,
        whitener: Whiten,
        extrinsics_sampler: Optional[
            Callable[[int], Tuple[BatchTensor, BatchTensor, BatchTensor]]
        ] = None,
        snr_rescaler: Optional[SnrRescaler] = None,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.whitener = whitener
        self.extrinsics_sampler = extrinsics_sampler
        self.snr_rescaler = snr_rescaler

    def forward(
        self,
        X: WaveformTensor,
        psds: Tensor,
        cross: Tensor,
        plus: Tensor,
        parameters: Optional[dict[str, Tensor]] = None,
        dec: Optional[BatchTensor] = None,
        psi: Optional[BatchTensor] = None,
        phi: Optional[BatchTensor] = None,
        target_snrs: Optional[BatchTensor] = None,
    ) -> Tuple[WaveformTensor, WaveformTensor, dict[str, Tensor], Optional[Tensor]]:
        """
        Args:
            X:
                Background timeseries to inject waveforms into, with
                shape ``(batch, num_ifos, time)``
            psds:
                Background PSDs to use for whitening after injection.
                Will be interpolated to match the number of frequency
                bins implied by ``X``'s length if necessary.
            cross:
                Cross polarizations of the waveforms to inject
            plus:
                Plus polarizations of the waveforms to inject
            parameters:
                Dictionary of waveform parameters to be updated in
                place with the (possibly sampled) extrinsic parameters
                used for injection. If left as ``None``, a new
                dictionary containing only the extrinsic parameters
                will be returned.
            dec:
                Declination of each source. If left as ``None``,
                will be sampled using ``self.extrinsics_sampler``.
            psi:
                Polarization angle of each source. If left as
                ``None``, will be sampled using
                ``self.extrinsics_sampler``.
            phi:
                Right ascension of each source relative to the
                geocenter. If left as ``None``, will be sampled
                using ``self.extrinsics_sampler``.
            target_snrs:
                Target network SNRs to rescale each waveform to.
                Ignored if ``self.snr_rescaler`` is ``None``. If
                ``self.snr_rescaler`` is set but ``target_snrs``
                is ``None``, a random permutation of the injected
                waveforms' current SNRs will be used as the target.
        Returns:
            The injected, whitened timeseries; the raw (post-rescale)
            interferometer responses that were injected; ``parameters``
            updated with the extrinsic parameters used for injection;
            and the network SNRs of the injected waveforms if
            ``self.snr_rescaler`` was provided, otherwise ``None``.
        """
        if dec is None or psi is None or phi is None:
            if self.extrinsics_sampler is None:
                raise ValueError(
                    "Must either pass dec, psi, and phi directly, or "
                    "construct Injector with an extrinsics_sampler"
                )
            dec, psi, phi = self.extrinsics_sampler(len(cross))

        responses = self.projector(dec, psi, phi, cross=cross, plus=plus)

        snrs = None
        if self.snr_rescaler is not None:
            responses, snrs = self.snr_rescaler(responses, target_snrs)

        num_freqs = X.size(-1) // 2 + 1
        if psds.size(-1) != num_freqs:
            psds = torch.nn.functional.interpolate(
                psds, size=(num_freqs,), mode="linear"
            )

        X = X + responses
        X = self.whitener(X, psds)

        parameters = dict(parameters) if parameters is not None else {}
        parameters.update({"dec": dec, "psi": psi, "phi": phi})
        return X, responses, parameters, snrs
