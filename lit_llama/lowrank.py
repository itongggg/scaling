from email.policy import strict
import torch
import torch.nn as nn
def proximal(
    M: torch.Tensor,
    mu: float,
    t: float,
    shape: tuple,
    step: int = 50,
    forced: bool = True
) -> torch.Tensor:
    """Proximal operator for nuclear norm.
    """
    Xc = torch.randn_like(M)
    P = torch.zeros_like(M)
    P[:shape[0], :shape[1]] = 1
    for _ in range(step): 
        Y = Xc - t * P * (Xc - M)
        U, S, V = torch.linalg.svd(Y)
        S = torch.clamp(torch.abs(S - t * mu), max=0)
        Xc = U @ torch.diag(S) @ V.T
        if torch.norm(P * (Xc - M)) < 1e-5:
            break
    if forced:
        Xc[:shape[0], :shape[1]] = M[:shape[0], :shape[1]]
    M = Xc
    return M

