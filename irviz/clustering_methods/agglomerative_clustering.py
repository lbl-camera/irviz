from sklearn.cluster import AgglomerativeClustering
import numpy as np
from irviz.utils.mapper import einops_data_mapper
from irviz.utils.mapper import multiset_mapper
def agglomerative_clustering(U,
                             mask,
                             n_clusters=5):
    """

    Parameters
    ----------
    U: An (C Nx Ny) shaped array that can contains Nones
    n_clusters: The number of clusters desired

    Returns: A cluster map of size (Nx,Ny), can contain Nones
    -------
    """

    shape = U.shape
    if mask is None:
        mask = np.ones(shape[1:]).astype(bool)

    pseudo_mask = np.ones((shape[0])).astype(bool)

    data_mapper_object = einops_data_mapper(U.shape, mask, pseudo_mask)
    U_flat = data_mapper_object.spectral_tensor_to_spectral_matrix(np.asarray(U))
    assert n_clusters > 1

    clustering = AgglomerativeClustering(n_clusters=n_clusters-1).fit(U_flat)
    labels = clustering.labels_.reshape(-1,1)
    cluster_map = data_mapper_object.matrix_to_tensor(labels)

    return cluster_map[0,...]

def multimap_aglomerative_clustering(Us, masks, n_clusters):
    shapes = []
    make_masks = False
    if masks is None:
        masks = []
        make_masks = True

    for U in Us:
        shape = U.shape
        shapes.append(shape)
        if make_masks:
            masks.append(np.ones(shape[1:]).astype(bool))
    pseudo_mask = np.ones(shapes[0][0]).astype(bool)
    mapobj = multiset_mapper(shapes, masks, pseudo_mask)
    U_flat = mapobj.spectral_tensor_to_spectral_matrix(Us)

    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(U_flat)
    labels = clustering.labels_.reshape(-1,1)
    cluster_map = mapobj.matrix_to_tensor(labels)
    return cluster_map







