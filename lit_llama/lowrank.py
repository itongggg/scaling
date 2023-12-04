from email.policy import strict
import torch

def proximal(
    X: torch.Tensor,
    P: torch.Tensor,
    M: torch.Tensor,
    mu: float,
    lam: float,
    t: float,
) -> torch.Tensor:
    """Proximal operator for nuclear norm.
    """
    X = X - t * P * (X - M)
    U, S, V = torch.linalg.svd(X)
    S = torch.clamp(S - mu * lam, min=0)
    return torch.mm(U, torch.mm(torch.diag(S), V.t()))