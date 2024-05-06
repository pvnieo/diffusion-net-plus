# 3p
import torch
# project
from .preprocess import compute_diffusion_operators
from .utils import sparse_np_to_sparse_tensor


class DiffusionOperatorsTransform(object):
    def __init__(self, n_eig=120, save_L=True, save_frames=False):
        self.n_eig = n_eig
        self.save_frames = save_frames
        self.save_L = save_L

    def __call__(self, data):
        verts, faces = data.pos, data.face.transpose(0, 1).contiguous()

        frames, mass, L, evals, evecs, gradX, gradY = compute_diffusion_operators(verts, faces, self.n_eig)

        # save data
        data.pos = verts.float()
        data.face = faces.transpose(0, 1).contiguous().long()
        data.mass = sparse_np_to_sparse_tensor(mass)
        data.evals = torch.from_numpy(evals).float()
        data.evecs = torch.from_numpy(evecs).float()
        data.gradX = sparse_np_to_sparse_tensor(gradX)
        data.gradY = sparse_np_to_sparse_tensor(gradY)

        if self.save_frames:
            data.frames = torch.from_numpy(frames).float()

        if self.save_L:
            data.L = sparse_np_to_sparse_tensor(L)

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(k_eig={self.n_eig}, save_L={self.save_L}, save_frames={self.save_frames})"
