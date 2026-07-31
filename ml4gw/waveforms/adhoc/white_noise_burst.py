import math
import torch
from torch import nn, Tensor
from ml4gw.types import BatchTensor
from .waveform_helper import turkey_window, semi_major_minor_from_e

class WhiteNoiseBurst(nn.Module):
    """
    Faithful PyTorch re-implementation of XLALGenerateBandAndTimeLimitedWhiteNoiseBurst.

    On each forward() call, new independent white-noise bursts are generated (one
    for h₊ and one for hₓ) following these steps:
      - Compute the time-series length: floor(k_len * duration / delta_t / 2)*2 + 1. (lal's implementation uses k_len=21)
      - Apply a time-domain Gaussian window with effective sigma = sqrt(duration²/4 - 1/(π² * bandwidth²)).
      - Transform to the frequency domain (rFFT).
      - Apply a frequency-domain Gaussian envelope centered at 'frequency' (with width = bandwidth/2),
        and adjust amplitudes with elliptical factors: a = √(1+eccentricity) for h₊, b = √(1–eccentricity) for hₓ.
      - For non-DC bins, rotate the phase by exp(–i·phase) for h₊ and by i·exp(–i·phase) for hₓ.
      - Normalize so that ∫(ḣ₊²+ḣₓ²)dt equals int_hdot_squared.
      - Inverse FFT back to the time domain and apply a final Tukey window (α=0.5) to smooth the edges.
    """

    def __init__(self, sample_rate: float, duration: float):
        """
        Args:
            sample_rate: Sampling rate in Hz.
            duration: Nominal burst duration in seconds.
            device: "cpu" or "cuda".
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
        frequency: BatchTensor,
        bandwidth: BatchTensor,
        eccentricity: BatchTensor,
        phase: BatchTensor,
        int_hdot_squared: BatchTensor,
        duration: BatchTensor
    ):
        """
        Generate a band- and time-limited white noise burst.

        Args:
            frequency: (batch,) Center frequency (Hz).
            bandwidth: (batch,) Frequency-domain 1-σ extent (Hz); Gaussian envelope has width = bandwidth/2.
            eccentricity: (batch,) Value in [0, 1] setting elliptical amplitude factors.
            phase: (batch,) Overall phase offset (radians).
            int_hdot_squared: (batch,) Desired ∫(ḣ₊² + ḣₓ²) dt.

        Returns:
            A tuple (h_cross, h_plus), each of shape (batch, length).
        """
        dtype = torch.float64
        device = frequency.device

        frequency = frequency.view(-1, 1).to(dtype=dtype)
        bandwidth = bandwidth.view(-1, 1).to(dtype=dtype)
        eccentricity = eccentricity.view(-1, 1).to(dtype=dtype)
        phase = phase.view(-1, 1).to(dtype=dtype)
        int_hdot_squared = int_hdot_squared.view(-1, 1).to(dtype=dtype)
        duration = duration.view(-1,1).to(dtype=dtype)

        batch = frequency.shape[0]
        length = self.length

        if (self.duration < 0) or (frequency < 0).any() or (bandwidth < 0).any() \
           or (eccentricity < 0).any() or (eccentricity > 1).any() or (int_hdot_squared < 0).any():
            raise ValueError("Invalid input parameters.")

        sigma_t_sq = (
            duration**2 / 4.0 
            - 1.0 
            / (
                torch.pi**2 
                * bandwidth**2
            )
        )

        if (sigma_t_sq < 0).any():
            raise ValueError("Invalid input parameters: sigma_t² < 0 (duration*bandwidth too small).")

        sigma_t = sigma_t_sq.sqrt()

        hplus = torch.randn(
            batch, 
            length, 
            device=device, 
            dtype=dtype
        )
        
        hcross = torch.randn(
            batch, 
            length, 
            device=device, 
            dtype=dtype
        )

        t_row = self.times.to(
            dtype=dtype,
            device=device
        ).unsqueeze(0)

        w_time = torch.exp(
            -0.5 * (t_row / sigma_t)**2
        )

        hplus = hplus * w_time
        hcross = hcross * w_time

        Hplus = torch.fft.rfft(
            hplus, 
            dim=-1
        )
        
        Hcross = torch.fft.rfft(
            hcross, 
            dim=-1
        )

        nfreq = Hplus.shape[-1]

        df = torch.tensor(
            self.sample_rate / length,
            device=device,
            dtype=dtype,
        )

        k = torch.arange(
            nfreq, 
            device=device, 
            dtype=dtype
        ).unsqueeze(0)

        f_array = k * df

        f_offset = f_array - frequency 

        beta = (
            -0.5 / ((bandwidth / 2.0)**2)
        )

        w_freq = torch.exp(
            (f_offset**2) * beta
        )

        a, b = semi_major_minor_from_e(
            eccentricity
        )

        Hplus = Hplus * (
            a * w_freq
        )

        Hcross = Hcross * (
            b * w_freq
        )

        pf = torch.exp(
            -1j * phase
        )

        non_dc = (
            torch.arange(
                nfreq,
                device=device,
            )
            != 0
        ).unsqueeze(0)
        
        Hplus = torch.where(
            non_dc,
            Hplus * pf, 
            Hplus
        )

        Hcross = torch.where(
            non_dc, 
            Hcross * (1j * pf), 
            Hcross
        )

        f_phys = f_array

        factor = (
            2 * torch.pi * f_phys
        )**2

        power_plus = (
            torch.sum(
                factor 
                * torch.abs(Hplus)**2, 
                dim=-1
            ) 
            * df
        )

        power_cross = (
            torch.sum(
                factor 
                * torch.abs(Hcross)**2, 
                dim=-1
            ) 
            * df
        )

        current_hdotsq = (
            power_plus 
            + power_cross 
        )

        eps = torch.finfo(dtype).tiny
        target_hdotsq = (
            int_hdot_squared.squeeze(-1)
        )

        norm_factor = torch.sqrt(
            current_hdotsq 
            / target_hdotsq.clamp(min=eps)
        )

        norm_factor = (
            norm_factor
            .clamp(min=eps)
            .unsqueeze(-1)
        )

        Hplus = Hplus / norm_factor
        Hcross = Hcross / norm_factor

        hplus_time = (
            torch.fft.irfft(
                Hplus, 
                n=length, 
                dim=-1
            ) 
            * self.sample_rate
        )

        hcross_time = (
            torch.fft.irfft(
                Hcross, 
                n=length, 
                dim=-1
            ) 
            * self.sample_rate
        )

        tw = turkey_window(
            length, 
            alpha=0.5, 
            device=device, 
            dtype=dtype
        ).unsqueeze(0)

        hplus_time = hplus_time * tw
        hcross_time = hcross_time * tw

        return hcross_time, hplus_time