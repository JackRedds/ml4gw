import torch

from ml4gw.types import BatchTensor


class MorletGabor(torch.nn.Module):
    """
    Callable class for generating Morlet-Gabor wavelet waveforms,
    matching the frequency-domain implementation in HyperWave
    (`hyperwave.detectors.waveforms.wavelets.morlet_gabor_fd`).

    The wavelet is evaluated analytically in the frequency domain
    (continuous-FT convention)

        tau    = Q / (2 pi f0)
        Psi(f) = (A tau sqrt(pi) / 2) e^{-2i pi f t0}
                 [ e^{+i phi0} e^{-pi^2 tau^2 (f-f0)^2}
                 + e^{-i phi0} e^{-pi^2 tau^2 (f+f0)^2} ]

    with the elliptical polarisation model

        h_plus  = Psi
        h_cross = epsilon * 1j * Psi

    and transformed to the time domain with HyperWave's `infft`
    normalisation (`irfft(...) * sample_rate`), so that in the time
    domain `h_plus ~ A exp(-(t-t0)^2 / tau^2) cos(2 pi f0 (t-t0) + phi0)`.

    Unlike HyperWave, where `t0` is measured from the start of the
    segment, `t0` here is the offset from the center of the time window.

    Args:
        sample_rate: Sample rate of waveform
        duration: Duration of waveform
    """

    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.N = int(round(duration * sample_rate))
        frequencies = torch.fft.rfftfreq(
            self.N, 1.0 / sample_rate, dtype=torch.float64
        )
        self.register_buffer("frequencies", frequencies)

    def frequency_domain(
        self,
        quality: BatchTensor,
        frequency: BatchTensor,
        amplitude: BatchTensor,
        phase: BatchTensor,
        shifts: BatchTensor,
        ellipticity: BatchTensor,
    ):
        """
        Frequency-domain plus and cross polarizations on the rfft grid,
        in the continuous-FT convention. See `forward` for arguments.

        Returns:
            Complex tensors of cross and plus polarizations
        """
        # compute in float64 to avoid precision loss in the
        # e^{-2i pi f t0} phase factor
        quality = quality.view(-1, 1).double()
        frequency = frequency.view(-1, 1).double()
        amplitude = amplitude.view(-1, 1).double()
        phase = phase.view(-1, 1).double()
        # convert offset from the window center to time from the start
        shifts = shifts.view(-1, 1).double() + self.duration / 2
        ellipticity = ellipticity.view(-1, 1).double()

        pi = torch.pi
        # frequency grid the spectrum is evaluated on, as opposed to
        # `frequency`, which is the per-waveform central frequency
        f = self.frequencies

        tau = quality / (2.0 * pi * frequency)
        prefactor = amplitude * tau * pi**0.5 / 2.0
        lobe_minus = torch.exp(-(pi**2) * tau**2 * (f - frequency) ** 2)
        lobe_plus = torch.exp(-(pi**2) * tau**2 * (f + frequency) ** 2)
        time_shift = torch.exp(-2j * pi * f * shifts)

        plus = (
            prefactor
            * time_shift
            * (
                torch.exp(1j * phase) * lobe_minus
                + torch.exp(-1j * phase) * lobe_plus
            )
        )
        cross = ellipticity * 1j * plus
        return cross, plus

    def forward(
        self,
        quality: BatchTensor,
        frequency: BatchTensor,
        amplitude: BatchTensor,
        phase: BatchTensor,
        shifts: BatchTensor,
        ellipticity: BatchTensor,
    ):
        """
        Generate time-domain Morlet-Gabor wavelets.

        Args:
            quality:
                Quality factor of the wavelet; sets
                `tau = quality / (2 pi frequency)`
            frequency:
                Central frequency of the wavelet in Hz
            amplitude:
                Strain amplitude of the wavelet
            phase:
                Phase of the wavelet
            t0:
                Central time of the wavelet in seconds,
                relative to the center of the buffer
            ellipticity:
                Ellipticity of the signal in [-1, 1]. Sets
                `h_cross = ellipticity * 1j * h_plus`
                in the frequency domain.
        Returns:
            Tensors of cross and plus polarizations
        """
        dtype = frequency.dtype
        cross, plus = self.frequency_domain(
            quality, frequency, amplitude, phase, shifts, ellipticity
        )
        cross = torch.fft.irfft(cross, n=self.N) * self.sample_rate
        plus = torch.fft.irfft(plus, n=self.N) * self.sample_rate
        return cross.to(dtype), plus.to(dtype)
