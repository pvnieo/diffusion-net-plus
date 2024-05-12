# code adapted from the original implementation of the DiffusionNet paper: https://github.com/nmwsharp/diffusion-net

# 3p
import numpy as np
import scipy
import scipy.sparse.linalg as sla
from sklearn.neighbors import KDTree
import potpourri3d as pp3d
import torch


def compute_diffusion_operators(verts, faces, k_eig):
    # convert to numpy float32
    if isinstance(verts, torch.Tensor):
        verts = verts.cpu().numpy().astype(np.float32)
        faces = faces.cpu().numpy().astype(np.int32) if faces is not None else None
    elif isinstance(verts, np.ndarray):
        verts = verts.astype(np.float32)
        faces = faces.astype(np.int32) if faces is not None else None
    else:
        raise ValueError("verts must be a `torch.Tensor` or `np.ndarray`")

    eps = 1e-8

    # build the scalar Laplacian matrix
    L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN values found in the Laplacian matrix")

    # compute the mass matrix
    massvec = pp3d.vertex_areas(verts, faces)
    massvec += eps * np.mean(massvec)
    if np.isnan(massvec).any():
        raise RuntimeError("NaN values found in the mass matrix")

    # Prepare matrices
    L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
    massvec = scipy.sparse.diags(massvec)
    eigs_sigma = eps

    # compute the eigendecomposition
    failcount = 0
    while True:
        try:
            # we would be happy here to lower tol or maxiter since we don't need these to be super precise,
            # but for some reason those parameters seem to have no effect
            evals, evecs = sla.eigsh(L_eigsh, k=k_eig, M=massvec, sigma=eigs_sigma)

            # clip off any eigenvalues that end up slightly negative due to numerical weirdness
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            break
        except Exception as e:
            print(e)
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp")
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10 ** failcount)

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # build gradient matrices
    # for meshes, we use the same edges as were used to build the Laplacian.
    frames = build_tangent_frames(verts, faces)
    edges = np.stack((inds_row, inds_col), axis=0)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    grad_mat = build_grad(verts, edges, edge_vecs)

    # split complex gradient in to two real sparse matrices
    gradX = np.real(grad_mat)
    gradY = np.imag(grad_mat)

    return frames, massvec, L, evals, evecs, gradX, gradY


def build_tangent_frames(verts, faces):
    V = verts.shape[0]

    # compute normals
    vert_normals = vertex_normals(verts, faces)  # (V,3)

    # find an orthogonal basis
    basis_cand1 = np.tile(np.array([1, 0, 0])[None, :], (V, 1))
    basis_cand2 = np.tile(np.array([0, 1, 0])[None, :], (V, 1))
    basis_1_far = np.abs(array_dot(vert_normals, basis_cand1)) < 0.9
    basisX = np.where(basis_1_far[:, None], basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = np.cross(vert_normals, basisX)
    frames = np.stack((basisX, basisY, vert_normals), axis=-2)
    if np.any(np.isnan(frames)):
        raise ValueError("NaN coordinates found during building the tangent frame! Must be very degenerate mesh!")
    return frames


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    normals = mesh_vertex_normals(verts, faces, n_neighbors_cloud)

    # if any are NaN, wiggle slightly and recompute
    bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
    if bad_normals_mask.any():
        bbox = np.amax(verts, axis=0) - np.amin(verts, axis=0)
        scale = np.linalg.norm(bbox) * 1e-4
        wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
        wiggle_verts = verts + bad_normals_mask * wiggle
        normals = mesh_vertex_normals(wiggle_verts, faces, n_neighbors_cloud)

    # if still NaN assign random normals (probably means unreferenced verts in mesh)
    bad_normals_mask = np.isnan(normals).any(axis=1)
    if bad_normals_mask.any():
        normals[bad_normals_mask, :] = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5)[bad_normals_mask, :]
        normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    if np.any(np.isnan(normals)):
        raise ValueError("NaN values found during the computation of normals!")
    return normals


def edge_tangent_vectors(verts, frames, edges):
    """
    Get tangent vector of edges in each local frame
    :param verts:
    :param frames:
    :param edges:
    :return:
    """
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]
    compX = array_dot(edge_vecs, basisX)
    compY = array_dot(edge_vecs, basisY)
    edge_tangent = np.stack((compX, compY), axis=-1)
    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    # build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges.shape[1]):
        tail_ind = edges[0, iE]
        tip_ind = edges[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.0

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix((data_vals, (row_inds, col_inds)), shape=(N, N)).tocsc()

    return mat


def mesh_vertex_normals(verts, faces, n_neighbors_cloud):
    if faces is None or faces.size == 0:  # point cloud
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts[neigh_inds, :]
        neigh_points = neigh_points - verts[:, np.newaxis, :]
        vertex_normals = neighborhood_normal(neigh_points)
    else:
        face_n = face_normals(verts, faces)
        vertex_normals = np.zeros_like(verts)
        for i in range(3):
            np.add.at(vertex_normals, faces[:, i], face_n)

        vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=-1, keepdims=True)

    return vertex_normals


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def face_normals(verts, faces, normalized=True):
    coords = verts[faces]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    raw_normal = np.cross(vec_A, vec_B)
    if normalized:
        return normalize(raw_normal)
    return raw_normal


def project_to_tangent(vecs, unit_normals):
    # Given (..., 3) vectors and normals, projects out any components of vecs
    # which lies in the direction of normals. Normals are assumed to be unit.
    dots = array_dot(vecs, unit_normals)
    return vecs - unit_normals * dots[..., None]


def array_dot(a, b):
    return np.sum(a * b, axis=-1)


def normalize(x, axis=-1, eps=1e-8):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if largest:
        raise ValueError("can't do largest with cpu_kd")

    # Build the tree
    kd_tree = KDTree(points_target)

    k_search = k + 1 if omit_diagonal else k
    _, neighbors = kd_tree.query(points_source, k=k_search)

    if omit_diagonal:
        # Mask out self element
        mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

        # make sure we mask out exactly one element in each row, in rare case of many duplicate points
        mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

        neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1] - 1))

    return neighbors
