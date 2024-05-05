# 3p
from torch_geometric.data import Data


class DiffusionData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["mass", "L", "gradX", "gradY"]:
            return (0, 1)
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
