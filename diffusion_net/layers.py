# code adapted from the original implementation of the DiffusionNet paper: https://github.com/nmwsharp/diffusion-net

# 3p
import torch
import torch.nn as nn
from torch_geometric.utils import unbatch
from pyg_lib.ops import grouped_matmul


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(self, layer_sizes, dropout=0.5, use_bn=True, activation=nn.ReLU, name="mlp"):
        super().__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2) == len(layer_sizes)

            if dropout > 0. and i > 0:
                self.add_module(f"{name}_layer_dropout_{i:03d}", nn.Dropout(dropout))

            # Affine map
            self.add_module(f"{name}_layer_{i:03d}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if use_bn:
                self.add_module(f"{name}_layer_bn_{i:03d}", nn.BatchNorm1d(layer_sizes[i + 1]))

            # non-linearity (but not on the last layer)
            if not is_last:
                self.add_module(f"{name}_act_{i:03d}", activation())


class LearnedTimeDiffusion(nn.Module):
    def __init__(self, C_inout, method="spectral", init_time=None, init_std=2.0):
        super().__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)

        if init_time is None:
            nn.init.constant_(self.diffusion_time, 0.0)
        else:
            assert isinstance(init_time, (int, float)), "`init_time` must be a scalar"
            nn.init.normal_(self.diffusion_time, mean=init_time, std=init_std)

    def forward(self, x, mass, evals, evecs, batch):
        bs = int(batch.max() + 1)
        neig = evals.size(0) // bs

        # todo do we need to do clipping here? do we need to remove do torch.no_grad? is abs ok (Nick didn't use it initially)?
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        # Transform to spectral
        x_spec = to_basis(x, evecs, mass, batch)

        # Diffuse
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex
        evecs = unbatch(evecs, batch, dim=0)
        x_diffuse_spec = x_diffuse_spec.split(neig)
        x_diffuse = torch.cat(grouped_matmul(evecs, x_diffuse_spec), dim=0)

        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super().__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


def to_basis(values, basis, massvec, batch):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (V, D)
      - basis: (V, K)
      - massvec: sparse (V, V)
    Outputs:
      - (B,K,D) transformed values
    """
    basis_T = (massvec.t() @ basis).T  # (K,V) @ (V,V) = (K,V)
    basis_T = unbatch(basis_T, batch, dim=1)  # [K x V1, K x V2, ...]
    values = unbatch(values, batch, dim=0)  # [V1 x D, V2 x D, ...]
    return torch.cat(grouped_matmul(basis_T, values), dim=0)
