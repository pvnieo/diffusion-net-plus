:rotating_light::construction: **WORK IN PROGRESS** :construction::rotating_light:

# DiffusionNet++ :speaking_head: :spider_web: :heavy_plus_sign::heavy_plus_sign:

[![Paper](https://img.shields.io/badge/Paper-arXiv-brightgreen)](https://arxiv.org/abs/XXXX.XXXXX)

> Add batching support to the original DiffusionNet implementation, better performance, more features, and simpler API.

---

## :negative_squared_cross_mark: ToDos

- [ ] Add documentation to all functions
- [ ] Add installation files for pip
- [ ] push the code to PyPI
- [ ] clean the code (style, comments, variable names, unused parts, legacy code, etc.)
- [ ] reduce the number of dependencies

## :construction_worker: Installation
This implementation requires Python >= 3.8, you can install it locally using the following commands:

```bash
python setup.py sdist bdist_wheel
pip install .
```


An installation method using PyPI will be available soon.
```bash
pip install diffusion-net
```

#### Conda Environment
You can create a conda environment that runs this code as follows:
```bash
conda create -n diffusion -y
conda activate diffusion
conda install -y python=3.8
# use the correct cuda version, e.g., `11.7` for cuda 11.7
conda install pytorch=2.2 torchvision torchaudio pytorch-cuda=${CUDA-VERSION} -c pytorch -c nvidia -y
conda install pyg -c pyg
conda install pytorch-sparse -c pyg
pip install git+https://github.com/pyg-team/pyg-lib.git
pip install trimesh potpourri3d
```

## :book: Usage

The DiffusionNet++ has a simple API that can be used to instantiate the model, prepare the data, and batch it using the PyG library. The following code snippet demonstrates how to use the DiffusionNet++ model to extract features from a batch that has shapes of different sizes.

```python
import numpy as np
import trimesh
import torch
from torch_geometric.data import Batch

from diffusion_net import DiffusionData, DiffusionOperatorsTransform, DiffusionNet

# load some meshes
mesh1 = trimesh.load("path/to/mesh1.obj")  # a mesh with N1 vertices
mesh2 = trimesh.load("path/to/mesh2.obj")  # a mesh with N2 vertices, N2 != N1

v1, f1 = np.array(mesh1.vertices), np.array(mesh1.faces)
v2, f2 = np.array(mesh2.vertices), np.array(mesh2.faces)

# create the data objects
data1 = DiffusionData(pos=torch.from_numpy(v1), face=torch.from_numpy(f1).T)
data2 = DiffusionData(pos=torch.from_numpy(v2), face=torch.from_numpy(f2).T)

# compute the diffusion operators
diffusion_transform = DiffusionOperatorsTransform(n_eig=97)  # compute the diffusion net operators with 97 eigenvalues
data1 = diffusion_transform(data1)
data2 = diffusion_transform(data2)

# create a batch
my_batch = Batch.from_data_list([data1, data2])
my_batch.x = my_batch.pos.clone()  # set the input features to the positions

# create the model and do a forward pass
diffusion_net = DiffusionNet(3, 69)  # input features are 3D positions, output features dimension is 69
output = diffusion_net(my_batch)
print(output.x.shape)
>>> torch.Size([(N1 + N2), 69])
```

## :chart_with_upwards_trend: Results

We reproduced the results of the original paper on the XX dataset. The results are detailed in the technical report [here](https://arxiv.org/abs/XXXX.XXXXX).


## :mortar_board: Citation
If you find this work useful in your research, please consider citing:
```bibtex
@inproceedings{mallet2024atomsurf,
    title={AtomSurf : Surface Representation for Learning on Protein Structures},
    author={Vincent Mallet and Souhaib Attaiki and Maks Ovsjanikov},
    year={2024},
    eprint={2309.16519},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```