import torch

def projector_from_Y(Y, eps=1e-4):
    # Y: [B,64,6]
    Bt = Y.transpose(-1, -2)                  # [B,6,64]
    G  = Bt @ Y                               # [B,6,6]
    I  = torch.eye(G.size(-1), device=Y.device, dtype=Y.dtype).expand_as(G)
    A  = G + eps * I                          # [B,6,6]
    # Solve A X = Bt  => X = A^{-1} Bt  (X: [B,6,64])
    X  = torch.linalg.solve(A, Bt)
    P  = Y @ X                                # [B,64,64]  = Y (Y^T Y + epsI)^{-1} Y^T
    return P

def subspace_projector_loss(Y_pred, U_gt, eps=1e-4):
    # U_gt: [B,64,6] (assumed to span the target subspace; if not perfectly orthonormal, still OK)
    P_pred = projector_from_Y(Y_pred, eps=eps)
    P_gt   = U_gt @ U_gt.transpose(-1, -2)
    return ((P_pred - P_gt) ** 2).mean()

def subspace_residual_loss(Y_pred, U_gt):
    P_gt = U_gt @ U_gt.transpose(-1, -2)      # [B,64,64]
    R    = Y_pred - (P_gt @ Y_pred)           # (I - P_gt)Y
    return (R ** 2).mean()
