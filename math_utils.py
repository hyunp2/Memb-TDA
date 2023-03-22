import numpy as np
from sklearn import metrics
from scipy import optimize
import warnings
import math
import torch
from scipy.linalg import sqrtm
import persim
from gudhi.wasserstein import wasserstein_distance
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from typing import *

class linear_sum_assignment(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost_matrix):
        matchi, matchj = optimize.linear_sum_assignment(cost_matrix)
        matchi, matchj = list(map(lambda inp: torch.from_numpy(inp), (matchi, matchj) ))
        ctx.save_for_backward(cost_matrix, matchi, matchj)
        ctx.mark_non_differentiable(matchi, matchj)
        return matchi, matchj
    
    @staticmethod
    def backward(ctx, gi, gj):
        cost_matrix, matchi, matchj = ctx.saved_tensors
        down_grad = torch.zeros_like(cost_matrix)
        down_grad[matchi, matchj] = 1.
        return down_grad

def wasserstein(dgm1, dgm2, matching=False):
    """
    Perform the Wasserstein distance matching between persistence diagrams.
    Assumes first two columns of dgm1 and dgm2 are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).

    See the `distances` notebook for an example of how to use this.

    Parameters
    ------------

    dgm1: Mx(>=2) 
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2) 
        array of birth/death paris for PD 2
    matching: bool, default False
        if True, return matching information and cross-similarity matrix

    Returns 
    ---------

    d: float
        Wasserstein distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)

    """

#     S = np.array(dgm1)
    S = dgm1.clone()
    M = min(S.shape[0], S.numel())
    if S.numel() > 0:
        S = S[torch.isfinite(S[:, 1]), :]
        if S.shape[0] < M:
            warnings.warn(
                "dgm1 has points with non-finite death times;"+
                "ignoring those points"
            )
            M = S.shape[0]
#     T = np.array(dgm2)
    T = dgm2.clone()
    N = min(T.shape[0], T.numel())
    if T.numel() > 0:
        T = T[torch.isfinite(T[:, 1]), :]
        if T.shape[0] < N:
            warnings.warn(
                "dgm2 has points with non-finite death times;"+
                "ignoring those points"
            )
            N = T.shape[0]

    if M == 0:
#         S = np.array([[0, 0]])
        S = dgm1.new_zeros((1,2))
        M = 1
    if N == 0:
#         T = np.array([[0, 0]])
        T = dgm2.new_zeros((1,2))
        N = 1
    # Compute CSM between S and dgm2, including points on diagonal
#     DUL = metrics.pairwise.pairwise_distances(S, T)
    DUL = torch.cdist(S,T)

    # Put diagonal elements into the matrix
    # Rotate the diagrams to make it easy to find the straight line
    # distance to the diagonal
    cp = np.cos(math.pi/4)
    sp = np.sin(math.pi/4)
    R = torch.tensor([[cp, -sp], [sp, cp]]).to(dgm1)
    S = S[:, 0:2].mm(R)
    T = T[:, 0:2].mm(R)
#     D = np.zeros((M+N, M+N))
    D = dgm1.new_zeros((M+N, M+N))
#     np.fill_diagonal(D, 0)
    D.fill_diagonal_(0)
    D[0:M, 0:N] = DUL
    UR = torch.tensor(float('inf')) * dgm1.new_ones((M, M))
    UR_numpy = UR.detach().cpu().numpy()
    S_numpy = S.detach().cpu().numpy()
    np.fill_diagonal(UR_numpy, S_numpy[:, 1])
#     UR.fill_diagonal_(S[:,1])
    UR.data.copy_(torch.from_numpy(UR_numpy).to(dgm1))
    D[0:M, N:N+M] = UR
    UL = torch.tensor(float('inf')) * dgm1.new_ones((N, N))
#     np.fill_diagonal(UL, T[:, 1])
    UL_numpy = UL.detach().cpu().numpy()
    T_numpy = T.detach().cpu().numpy()
    np.fill_diagonal(UL_numpy, T_numpy[:, 1])
    UL.data.copy_(torch.from_numpy(UL_numpy).to(dgm1))
    D[M:N+M, 0:N] = UL

    # Step 2: Run the hungarian algorithm
#     matchi, matchj = optimize.linear_sum_assignment(D)
    matchi, matchj = linear_sum_assignment.apply(D)
    matchdist = D[matchi, matchj].sum()

    return matchdist

# https://github.com/ku-milab/LEAR/blob/e3b087ea100cc84b7f8da541cbd2e084284013c7/MNIST/M_test.py#L7
def calculate_fid(act1, act2):
    mu1, sigma1 = np.mean(act1), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def wasserstein_difference(args: argparse.ArgumentParser, temp0_dgms: List[np.array], temp1_dgms: List[np.array]):
    ...

if __name__ == "__main__":
    x = torch.randn(100,2).double().data
    x.requires_grad = True
    y = torch.randn(30, 2).double().data
    z = wasserstein(x, y)
    print(z)
    z.register_hook(lambda grad: grad)
    z.retain_grad()
    z.backward()
    print(x.grad)
    print(z.grad)
    print(torch.autograd.gradcheck(wasserstein, (x, y)))
