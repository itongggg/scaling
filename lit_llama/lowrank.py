import torch
from loguru import logger
def proximal(
    M: torch.Tensor,
    mu: float,
    t: float,
    shape: tuple,
    step: int = 200,
    reduction: bool = False
) -> torch.Tensor:
    """Proximal operator for nuclear norm.
    """

    # Xc = torch.randn_like(M)
    Xc = M.clone()
    P = torch.zeros_like(M)
    P[:shape[0], :shape[1]] = 1
    dist = 0.0
    for _ in range(step): 
        Y = Xc - t * P * (Xc - M)
        U, S, V = torch.linalg.svd(Y, full_matrices=False)
        S = torch.max(S - t*mu, torch.tensor(0.0))
        Xc = U @ torch.diag(S) @ V
    dist = torch.dist(P*Xc, P*M) / (torch.sum(P).item())
    logger.info("dist: {}", dist)
    if reduction:
        U, S, V = torch.linalg.svd(Xc, full_matrices=False)
        S = torch.where(S > 1e-6, S, torch.tensor(0.0))
        Xc = U @ torch.diag(S) @ V
    M = Xc
    return M

