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
    dist = 0.0
    for _ in range(step): 
        Y = Xc - t * P * (Xc - M)
        U, S, V = torch.linalg.svd(Y, full_matrices=False)
        S = torch.clamp(torch.abs(S - t * mu), max=0)
        # print(f"U: {U.shape}, S: {S.shape},  V: {V.shape}")
        Xc = U @ torch.diag(S) @ V
        dist = torch.dist(P*Xc, P*M)
        
        if dist < 1e-6:
            break
    # print("diff: ", torch.norm(P*(Xc - M)))
    # print(f"dist {dist}")
    if forced:
        Xc[:shape[0], :shape[1]] = M[:shape[0], :shape[1]]
    M = Xc
    return M

