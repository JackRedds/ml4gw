import torch
from torch import Tensor

from typing import Dict, Tuple
from ml4gw.types import BatchTensor


def semi_major_minor_from_e(e: Tensor):
    a = 1.0 / torch.sqrt(2.0 - (e * e))
    b = a * torch.sqrt(1.0 - (e * e))
    return a, b


class SineGaussian(torch.nn.Module):
    """
    Callable class for generating sine-Gaussian waveforms.

    Args:
        sample_rate: Sample rate of waveform
        duration: Duration of waveform
    """

    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        # determine times based on requested duration and sample rate
        # and shift so that the waveform is centered at t=0

        num = int(duration * sample_rate)
        times = torch.arange(num, dtype=torch.float64) / sample_rate
        times -= duration / 2.0

        self.register_buffer("times", times)

    def forward(
        self,
        quality: BatchTensor,
        frequency: BatchTensor,
        hrss: BatchTensor,
        phase: BatchTensor,
        eccentricity: BatchTensor,
    ):
        """
        Generate lalinference implementation of a sine-Gaussian waveform.
        See
        git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/lib/LALInferenceBurstRoutines.c#L381
        for details on parameter definitions.

        Args:
            frequency:
                Central frequency of the sine-Gaussian waveform
            quality:
                Quality factor of the sine-Gaussian waveform
            hrss:
                Hrss of the sine-Gaussian waveform
            phase:
                Phase of the sine-Gaussian waveform
            eccentricity:
                Eccentricity of the sine-Gaussian waveform.
                Controls the relative amplitudes of the
                hplus and hcross polarizations.
        Returns:
            Tensors of cross and plus polarizations
        """
        dtype = frequency.dtype
        # add dimension for calculating waveforms in batch
        frequency = frequency.view(-1, 1)
        quality = quality.view(-1, 1)
        hrss = hrss.view(-1, 1)
        phase = phase.view(-1, 1)
        eccentricity = eccentricity.view(-1, 1)

        # TODO: enforce all inputs are on the same device?
        pi = torch.tensor([torch.pi], device=frequency.device)

        # calculate relative hplus / hcross amplitudes based on eccentricity
        # as well as normalization factors
        a, b = semi_major_minor_from_e(eccentricity)
        norm_prefactor = quality / (4.0 * frequency * torch.sqrt(pi))
        cosine_norm = norm_prefactor * (1.0 + torch.exp(-quality * quality))
        sine_norm = norm_prefactor * (1.0 - torch.exp(-quality * quality))

        cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)

        h0_plus = (
            hrss
            * a
            / torch.sqrt(
                cosine_norm * (cos_phase**2) + sine_norm * (sin_phase**2)
            )
        )
        h0_cross = (
            hrss
            * b
            / torch.sqrt(
                cosine_norm * (sin_phase**2) + sine_norm * (cos_phase**2)
            )
        )

        # cast the phase to a complex number
        phi = 2 * pi * frequency * self.times
        complex_phase = torch.complex(torch.zeros_like(phi), (phi - phase))

        # calculate the waveform and apply a tukey window to taper the waveform
        fac = torch.exp(phi**2 / (-2.0 * quality**2) + complex_phase)

        cross = fac.imag * h0_cross
        plus = fac.real * h0_plus

        cross = cross.to(dtype)
        plus = plus.to(dtype)

        return cross, plus


class MultiSineGaussian(SineGaussian):
    def __init__(self, sample_rate: float, duration: float, max_shift: float = 1e-3):
        super().__init__(sample_rate=sample_rate, duration=duration)
        self.max_shift = max_shift
        self.sample_rate = sample_rate
        self.duration = duration

    def ave_parameters(self, parameters: Dict[str, torch.Tensor]):
        averaged_params = {}
        for i, params in parameters.items():
            for k, v in params.items():
                if k not in averaged_params:
                    averaged_params[k] = []
                averaged_params[k].append(v.mean(dim=0))
        # average parameters
        for k in averaged_params:
            averaged_params[k] = torch.stack(averaged_params[k])
        return averaged_params
    
    def shift_waveforms(self, cross: torch.Tensor, plus: torch.Tensor, shifts: torch.Tensor):
        N = cross.shape[0]
        shift_samples = (shifts * self.sample_rate).long()
        shifted_cross = torch.zeros_like(cross)
        shifted_plus = torch.zeros_like(plus)
        for i in range(N):
            shift = shift_samples[i].item()
            if shift > 0:
                shifted_cross[i, shift:] = cross[i, :-shift]
                shifted_plus[i, shift:] = plus[i, :-shift]
            elif shift < 0:
                shifted_cross[i, :shift] = cross[i, -shift:]
                shifted_plus[i, :shift] = plus[i, -shift:]
            else:
                shifted_cross[i] = cross[i]
                shifted_plus[i] = plus[i]
        return shifted_cross, shifted_plus
    
    def forward(self, **parameters):
        cross_waveforms = []
        plus_waveforms = []
        for i, params in parameters.items():
            shifts = params["shifts"]
            params = {
                k: v for k, v in params.items() if k != "shifts"
            }
            cross, plus = super().forward(**params)
            cross, plus = self.shift_waveforms(cross, plus, shifts)
            cross = cross.mean(dim=0, keepdim=True)
            plus = plus.mean(dim=0, keepdim=True)
            cross_waveforms.append(cross)
            plus_waveforms.append(plus)
        cross = torch.vstack(cross_waveforms)
        plus = torch.vstack(plus_waveforms)

        return cross, plus