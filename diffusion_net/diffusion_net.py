# code adapted from the original implementation of the DiffusionNet paper: https://github.com/nmwsharp/diffusion-net

# 3p
import torch
import torch.nn as nn
# project
from .layers import LearnedTimeDiffusion, MiniMLP, SpatialGradientFeatures


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims=None, n_layers=2, dropout=0.5, use_bn=True, init_time=2.0, init_std=2.0,
                 diffusion_method="spectral", with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()

        # Specified dimensions
        self.C_width = C_width
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width] * n_layers
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method, init_time=init_time,
                                              init_std=init_std)

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                self.C_width, with_gradient_rotations=self.with_gradient_rotations
            )
            self.MLP_C += self.C_width

        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + list(self.mlp_hidden_dims) + [self.C_width],
                           dropout=self.dropout,
                           use_bn=use_bn)

        # todo: is this needed?
        # self.bn = nn.BatchNorm1d(C_width)

    def forward(self, surfaces):
        x_in = surfaces.x

        # Diffusion block
        x_diffuse = self.diffusion(x_in, surfaces.mass, surfaces.evals, surfaces.evecs, surfaces.batch)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_gradX = surfaces.gradX @ x_diffuse
            x_gradY = surfaces.gradY @ x_diffuse
            x_grad = torch.stack((x_gradX, x_gradY), dim=-1)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad)

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        # # apply batch norm # todo: is this needed?
        # x0_out_batch = self.bn(x0_out_batch)

        # update the features
        surfaces.x = x0_out
        return surfaces


class DiffusionNet(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, mlp_hidden_dims=None, dropout=0.5,
                 with_gradient_features=True, with_gradient_rotations=True, use_bn=True, init_time=2.0, init_std=2.0):

        super().__init__()

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation

        # MLP options
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # # Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(
                C_width=C_width,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                use_bn=use_bn,
                init_time=init_time,
                init_std=init_std
            )

            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(self, surface):
        surface.x = self.first_lin(surface.x)

        # Apply each of the blocks
        for block in self.blocks:
            surface = block(surface)

        # Apply the last linear layer
        surface.x = self.last_lin(surface.x)

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            surface.x = self.last_activation(surface.x)

        return surface
