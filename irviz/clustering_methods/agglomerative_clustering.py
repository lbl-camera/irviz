from sklearn.cluster import AgglomerativeClustering
import numpy as np
import einops

def agglomarativeclustering(U,
                            n_clusters):
    """

    Parameters
    ----------
    U: An (C Nx Ny) shaped array that can contains Nones
    n_clusters: The number of clusters desired

    Returns: A cluster map of size (Nx,Ny), can contain Nones
    -------
    """
    assert n_clusters > 1

    U_flat = einops.rearrange(U, "C Nx Ny -> (Nx Ny) C" )
    sel = ~np.isnan(U_flat)
    sel = np.sum(sel, axis=1).astype(bool)

    U_data = U_flat[sel, :]
    # do the clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(U_data)
    cluster_map = np.zeros( (U.shape[1], U.shape[2]) )
    cluster_map = einops.rearrange(cluster_map, "Nx Ny -> (Nx Ny)")

    # fill in pixels we didn't use with Nones
    cluster_map[ ~sel ] = None
    # place labels
    cluster_map[sel] = clustering.labels_
    # reshape
    cluster_map = einops.rearrange(cluster_map, "(Nx Ny) -> Nx Ny", Nx=U.shape[1], Ny=U.shape[2] )
    return cluster_map