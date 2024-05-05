# 3p
import numpy as np
import torch
from torch_sparse import SparseTensor


def sparse_np_to_sparse_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()


def sparse_np_to_sparse_tensor(A):
    sparse_torch = sparse_np_to_sparse_torch(A)
    return SparseTensor.from_torch_sparse_coo_tensor(sparse_torch.float())
