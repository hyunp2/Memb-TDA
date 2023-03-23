import numpy as np
from sklearn import metrics
from scipy import optimize
import warnings
import math
import torch
from scipy.linalg import sqrtm
import persim
# from gudhi.wasserstein import wasserstein_distance
from persim import wasserstein as wasserstein_distance
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from typing import *
# from persim import plot_diagrams
from visual_utils import plot_diagrams
import collections
import matplotlib.pyplot as plt

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


def wasserstein_matching(dgm1, dgm2, matching, original_dgms1, original_dgms2, labels=["dgm1", "dgm2", "original_dgms1", "original_dgms2"], ax=None):
    """ 
    https://persim.scikit-tda.org/en/latest/_modules/persim/visuals.html#wasserstein_matching
    
    Visualize bottleneck matching between two diagrams

    Parameters
    ===========

    dgm1: array
        A diagram
    dgm2: array
        A diagram
    matching: ndarray(Mx+Nx, 3)
        A list of correspondences in an optimal matching, as well as their distance, where:
        * First column is index of point in first persistence diagram, or -1 if diagonal
        * Second column is index of point in second persistence diagram, or -1 if diagonal
        * Third column is the distance of each matching
    labels: list of strings
        names of diagrams for legend. Default = ["dgm1", "dgm2"], 
    ax: matplotlib Axis object
        For plotting on a particular axis.

    Examples
    ==========

    bn_matching, (matchidx, D) = persim.wasserstien(A_h1, B_h1, matching=True)
    persim.wasserstein_matching(A_h1, B_h1, matchidx, D)

    """
    ax = ax or plt.gca()

    cp = np.cos(np.pi / 4)
    sp = np.sin(np.pi / 4)
    R = np.array([[cp, -sp], [sp, cp]])
    if dgm1.size == 0:
        dgm1 = np.array([[0, 0]])
    if dgm2.size == 0:
        dgm2 = np.array([[0, 0]])
    dgm1Rot = dgm1.dot(R)
    dgm2Rot = dgm2.dot(R)
    for [i, j, d] in matching:
        i = int(i)
        j = int(j)
        if i != -1 or j != -1: # At least one point is a non-diagonal point
            if i == -1:
                diagElem = np.array([dgm2Rot[j, 0], 0])
                diagElem = diagElem.dot(R.T)
                plt.plot([dgm2[j, 0], diagElem[0]], [dgm2[j, 1], diagElem[1]], 'tab:green', alpha=0.5)
            elif j == -1:
                diagElem = np.array([dgm1Rot[i, 0], 0])
                diagElem = diagElem.dot(R.T)
                ax.plot([dgm1[i, 0], diagElem[0]], [dgm1[i, 1], diagElem[1]], 'tab:green', alpha=0.5)
            else:
                ax.plot([dgm1[i, 0], dgm2[j, 0]], [dgm1[i, 1], dgm2[j, 1]], 'tab:red', alpha=1.)

#     fig, ax = plt.subplots(1,1)
    plot_diagrams([dgm1, dgm2], labels=labels[:2], ax=ax, c=["tab:blue", "tab:orange"], marker="X", size=30, show=False, save=None)
    plot_diagrams([np.concatenate(original_dgms1, axis=0), np.concatenate(original_dgms2, axis=0)], size=13, labels=labels[2:], alpha=0.4, c=["tab:blue", "tab:orange"], marker="o", ax=ax, show=False, save="wass.png")

def wasserstein_difference(temp0_dgms: List[np.array], temp1_dgms: List[np.array]):
    wass = collections.namedtuple('wass', ['barycenter0', 'barylog0', 'barycenter1', 'barylog1', 'wdist', 'windex'])
    if isinstance(temp0_dgms, np.ndarray): temp0_dgms = [temp0_dgms]
    if isinstance(temp1_dgms, np.ndarray): temp1_dgms = [temp0_dgms]
    assert isinstance(temp0_dgms, list) and isinstance(temp1_dgms, list), "Both instances should be a list!"
    barycenter0, barylog0 = lagrangian_barycenter(temp0_dgms, verbose=True)
    barycenter1, barylog1 = lagrangian_barycenter(temp1_dgms, verbose=True)
    
    wdist, windex = wasserstein_distance(barycenter0, barycenter1, matching=True)
#     print(barycenter1.shape, barycenter0.shape, windex)
    wasserstein_matching(barycenter0, barycenter1, windex, temp0_dgms, temp1_dgms, labels=['lower temp', 'higher temp', "all lower temps", "all higher temps"]) #plot
    [setattr(wass, key, val) for key, val in zip(['barycenter0', 'barylog0', 'barycenter1', 'barylog1', 'wdist', 'windex'], [barycenter0, barylog0, barycenter1, barylog1, wdist, windex])]
    return wass

if __name__ == "__main__":
#     x = torch.randn(100,2).double().data
#     x.requires_grad = True
#     y = torch.randn(30, 2).double().data
#     z = wasserstein(x, y)
#     print(z)
#     z.register_hook(lambda grad: grad)
#     z.retain_grad()
#     z.backward()
#     print(x.grad)
#     print(z.grad)
#     print(torch.autograd.gradcheck(wasserstein, (x, y)))

    wass = wasserstein_difference([np.random.randn(100,2), np.random.randn(100,2)], [np.random.randn(60,2), np.random.randn(100,2)])
    print(wass.wdist)
