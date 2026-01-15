import torch
import torch.nn.functional as F

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

import torch
import torch.nn.functional as F

def projection_align_loss(Y_pred, U_gt, eps=1e-6):
    """
    Y_pred: [B, 64, 6]  (network output)
    U_gt  : [B, 64, 6]  (ground-truth left singular vectors or any orthonormal basis of target subspace)
    """
    # Orthonormalize prediction via thin QR (no SVD)
    Q_pred, _ = torch.linalg.qr(Y_pred, mode='reduced')     # [B,64,6]

    # If U_gt might not be perfectly orthonormal, re-orthonormalize too
    Q_gt, _ = torch.linalg.qr(U_gt, mode='reduced')         # [B,64,6]

    P_pred = Q_pred @ Q_pred.transpose(-1, -2)              # [B,64,64]
    P_gt   = Q_gt   @ Q_gt.transpose(-1, -2)                # [B,64,64]

    # Frobenius distance between projectors
    return ((P_pred - P_gt) ** 2).mean()

def grassmann_loss(Y_pred, U_gt):
    Q_pred, _ = torch.linalg.qr(Y_pred, mode='reduced')  # [B,64,6]
    Q_gt, _   = torch.linalg.qr(U_gt, mode='reduced')    # [B,64,6]

    M = Q_pred.transpose(-1, -2) @ Q_gt                 # [B,6,6]
    # Want M close to orthogonal => large Fro norm
    r = M.shape[-1]
    return (r - (M ** 2).sum(dim=(-1, -2))).mean()
