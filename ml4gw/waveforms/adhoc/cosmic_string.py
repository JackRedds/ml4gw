import math
import torch
from torch import Tensor
from ml4gw.types import BatchTensor, Tensor
from .turkey_window import turkey_window

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
        power: float,
        amplitude: BatchTensor,
        f_high: float
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
        amplitude = amplitude.view(-1, 1)
        batch = amplitude.shape[0]

        device = amplitude.device
        f_low = 1.0

        length = self.length
        dt = 1 / self.sample_rate
        freq_bins = length // 2 + 1

        freq = torch.fft.rfftfreq(length, d=dt, device=device).unsqueeze(0)
        k = torch.arange(freq_bins, dtype=torch.float64, device=device)
        phase_factor = torch.exp(-1j * math.pi * k * (length - 1) / float(length))

        Hf = torch.zeros((batch, freq_bins), dtype=torch.complex128, device=device)

        valid_mask = torch.ones(freq_bins, dtype=torch.bool, device=device)
        valid_mask[0] = False
        valid_mask[-1] = False
        valid_mask_2d = valid_mask.unsqueeze(0)

        f_clamped = torch.clamp(freq, min=1e-20)

        base_factor = (1.0 + (f_low**2) / (f_clamped**2))**(-4.0)

        base_factor *= f_clamped**(power)

        ratio = freq / f_high 
        taper = torch.where(ratio > 1.0, torch.exp(1.0 - ratio), torch.ones_like(ratio))

        amp_val = amplitude * base_factor
        amp_val = amp_val * taper

        amp_val = torch.where(valid_mask_2d, amp_val, torch.zeros_like(amp_val))

        A = amp_val * phase_factor

        hplus = torch.fft.irfft(A, n=length, dim=-1)
        hplus = hplus*self.sample_rate
        hcross = torch.zeros_like(hplus)

        tw = turkey_window(length, alpha=0.5, device=device, dtype=torch.float64)
        hplus = hplus * tw

        return hcross, hplus