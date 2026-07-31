import torch
from torch import Tensor
from ml4gw.types import BatchTensor
from .waveform_helper import turkey_window, semi_major_minor_from_e

class Gaussian(torch.nn.Module):
    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        num = int(duration * sample_rate)
        self.length = num
        times = torch.arange(num, dtype=torch.float64) / sample_rate
        times -= duration / 2.0

        self.register_buffer("times", times)


    def forward(
            self,
            hrss: BatchTensor, 
            polarization: BatchTensor, 
            eccentricity: BatchTensor, 
            duration: BatchTensor
        ):

        hrss       = hrss.view(-1, 1)
        psi        = polarization.view(-1, 1)
        duration = duration.view(-1, 1)
        device = hrss.device
        dtype = hrss.dtype

        # correct LAL normalisation
        h0 = (
            hrss 
            / torch.sqrt(
                torch.sqrt(
                    torch.tensor(
                        torch.pi,
                        dtype=dtype, 
                        device=device
                    )
                ) 
                * self.length
            )
        )

        h0_plus  = h0 * torch.cos(psi)
        h0_cross = h0 * torch.sin(psi)

        t = self.times.to(device, dtype)
        env = torch.exp(
            -0.5 * t.pow(2).view(1,-1) / duration**2
        )

        alpha = 0.5
        tw = turkey_window(
            t.shape[-1], 
            alpha, 
            device=device, 
            dtype=dtype
        )

        plus = h0_plus * env
        cross = h0_cross * env

        plus  = (h0_plus  * env) * tw
        cross = (h0_cross * env) * tw
        return cross, plus