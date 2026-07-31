import math
import torch
from torch import Tensor
from ml4gw.types import BatchTensor, Tensor
from .waveform_helper import turkey_window, semi_major_minor_from_e

class CosmicString(torch.nn.Module):
    """
    PyTorch re-implementation of the 'XLALGenerateString' logic for cosmic-string
    waveforms: 'cusp', 'kink', or 'kinkkink'.
    LAL sets:
      - f_low = 1 Hz
      - length = floor(9.0 / f_low / dt / 2) * 2 + 1  => ~9 seconds total
      - The waveforms are built in frequency domain and iFFT to time domain.
    """

    def __init__(self, sample_rate: float, duration: float):
        """
        Args:
            sample_rate: sampling rate (Hz).
            duration: duration of the waveform in seconds.
            device: which device ("cpu" or "cuda") to store buffers on.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration

        num = int(duration * sample_rate)
        self.length = num
        times = torch.arange(num, dtype=torch.float64) / sample_rate
        times -= duration / 2.0

        self.register_buffer("times", times)

    def forward(
        self,
        power: BatchTensor,
        amplitude: BatchTensor,
        f_high: BatchTensor,
    ):
        """
        Generate the chosen cosmic-string waveform in plus polarization,
        with cross=0.
        waveform must be: "cusp"  -4.0 / 3.0, "kink" -5.0 / 3.0, or "kinkkink" -2.0.
        Args:
            power: cusp = -4.0 / 3.0, kink = -5.0 / 3.0, or kinkkink = -2.0.
            amplitude: (batch,) overall amplitude scaling parameter.
            f_high: (batch,) freq above which we apply exponential taper.
        Returns:
            (h_cross, h_plus): shape (batch, self.length).
            The cross polarization is zero (as in LAL).
        """

        # ---------------------------------------------------------
        # Make all parameters explicitly (batch, 1)
        # ---------------------------------------------------------

        power = power.view(-1, 1)
        amplitude = amplitude.view(-1, 1)
        f_high = f_high.view(-1, 1)
        batch = amplitude.shape[0]

        # ---------------------------------------------------------
        # Basic setup
        # ---------------------------------------------------------

        device = amplitude.device
        dtype = amplitude.dtype
        f_low = 1.0

        length = self.length
        dt = 1 / self.sample_rate
        freq_bins = length // 2 + 1

        freq = torch.fft.rfftfreq(
            length, 
            d=dt, 
            device=device
        ).to(dtype).unsqueeze(0)

        # ---------------------------------------------------------
        # Phase factor
        # ---------------------------------------------------------

        k = torch.arange(
            freq_bins, 
            dtype=dtype,
            device=device
        )

        phase_factor = torch.exp(
            -1j * math.pi * k * (length - 1) / float(length)
        )

        phase_factor = phase_factor.unsqueeze(0)

        # ---------------------------------------------------------
        # Frequency mask
        # ---------------------------------------------------------

        valid_mask = torch.ones(
            freq_bins, 
            dtype=torch.bool, 
            device=device
        )

        valid_mask[0] = False
        valid_mask[-1] = False

        valid_mask = valid_mask.unsqueeze(0)

        # ---------------------------------------------------------
        # Avoid division by zero at f = 0
        # ---------------------------------------------------------

        f_clamped = torch.clamp(
            freq, 
            min=1e-20
        )

        # ---------------------------------------------------------
        # Cosmic-string power-law spectrum
        # ---------------------------------------------------------

        base_factor = (1.0 + (f_low**2) / (f_clamped**2))**(-4.0)

        base_factor = base_factor * (f_clamped ** power)

        # ---------------------------------------------------------
        # High-frequency exponential taper
        # ---------------------------------------------------------

        ratio = freq / f_high 

        taper = torch.where(
            ratio > 1.0, 
            torch.exp(1.0 - ratio), 
            torch.ones_like(ratio)
        )

        # ---------------------------------------------------------
        # Apply amplitude
        # ---------------------------------------------------------

        amp_val = amplitude * base_factor
        amp_val = amp_val * taper

        # ---------------------------------------------------------
        # Remove DC and Nyquist bins
        # ---------------------------------------------------------

        amp_val = torch.where(
            valid_mask,
            amp_val, 
            torch.zeros_like(amp_val)
        )

        # ---------------------------------------------------------
        # Apply phase
        # ---------------------------------------------------------

        A = amp_val * phase_factor

        # ---------------------------------------------------------
        # Frequency -> time domain
        # ---------------------------------------------------------

        hplus = torch.fft.irfft(
            A, 
            n=length, 
            dim=-1
        )

        hplus = hplus * self.sample_rate

        hcross = torch.zeros_like(hplus)

        # ---------------------------------------------------------
        # Turkey window
        # ---------------------------------------------------------

        tw = turkey_window(
            length, 
            alpha=0.5, 
            device=device, 
            dtype=dtype
        )

        hplus = hplus * tw

        return hcross, hplus