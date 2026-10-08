import math
import torch
from torch import Tensor


def tukey_window(n: int, alpha: float = 0.5, device=None, dtype=None):
    """
    Generate a symmetric length-n Tukey window, matching
    scipy.signal.windows.tukey and LAL's XLALCreateTukeyREAL8Window.
    alpha is the fraction of the window inside the cosine tapers:
    alpha=0.5 => 25% of the samples on each end are tapered, 50% are
    flat in the middle. alpha <= 0 is rectangular, alpha >= 1 is Hann.
    """
    w = torch.ones(n, device=device, dtype=dtype)
    if alpha <= 0 or n < 2:
        return w
    alpha = min(alpha, 1.0)
    edge = alpha / 2
    x = torch.arange(n, device=device, dtype=dtype) / (n - 1)
    rise = 0.5 * (1.0 - torch.cos(torch.pi * x / edge))
    fall = 0.5 * (1.0 - torch.cos(torch.pi * (1.0 - x) / edge))
    w = torch.where(x < edge, rise, w)
    w = torch.where(x > 1.0 - edge, fall, w)
    return w

def semi_major_minor_from_e(e: Tensor):
    a = 1.0 / torch.sqrt(2.0 - (e * e))
    b = a * torch.sqrt(1.0 - (e * e))
    return a, b