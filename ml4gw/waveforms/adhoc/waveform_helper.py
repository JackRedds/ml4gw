import math
import torch
from torch import Tensor


def turkey_window(n: int, alpha: float = 0.5, device=None, dtype=None):
    """
    Generate a length-n Tukey window with fraction alpha of the window tapered.
    alpha=0.5 => 25% of the samples on each end are tapered, 50% are flat in the middle.
    """
    w = torch.ones(n, device=device, dtype=dtype)
    if alpha <= 0:
        return w
    if alpha >= 1:
        t_lin = torch.linspace(0, torch.pi, n, device=device, dtype=dtype)
        return 0.5 * (1.0 - torch.cos(t_lin))
    taper_len = int(alpha * (n - 1) / 2)
    t1 = torch.linspace(0, torch.pi / 2, taper_len, device=device, dtype=dtype)
    w[:taper_len] = 0.5 * (1.0 - torch.cos(t1))
    t2 = torch.linspace(torch.pi / 2, 0, taper_len, device=device, dtype=dtype)
    w[-taper_len:] = 0.5 * (1.0 - torch.cos(t2))
    return w

def semi_major_minor_from_e(e: Tensor):
    a = 1.0 / torch.sqrt(2.0 - (e * e))
    b = a * torch.sqrt(1.0 - (e * e))
    return a, b