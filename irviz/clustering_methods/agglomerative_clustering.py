from sklearn.cluster import AgglomerativeClustering
import numpy as np
import einops

from sklearn.neighbors import LocalOutlierFactor

def agglomerative_clustering(U,
                             mask,
                             n_clusters=5,
                             outlier_neighbours=3,
                             outlier_p=2,
                             outlier_contamination=0.005):
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
    original_indices = np.arange(sel.shape[0])
    U_data = U_flat[sel, :]
    first_pass = original_indices[sel]

    # first we find outliers
    if outlier_contamination < 0:
        outlier_detector = LocalOutlierFactor(n_neighbors=outlier_neighbours,
                                                            algorithm='auto',
                                                            leaf_size=30,
                                                            metric='minkowski',
                                                            p=outlier_p,
                                                            metric_params=None,
                                                            contamination=outlier_contamination,
                                                            novelty=False,
                                                            n_jobs=None)
        outliers = outlier_detector.fit_predict(U_data)
        sel2 = outliers < 0
    else:
        sel2 = np.zeros( sel.shape )
    second_pass = first_pass[~sel2]
    outlier_indices = first_pass[sel2]

    culled_data = U_data[~sel2,:]
    # do the clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters-1).fit(culled_data)
    cluster_map = np.zeros( (U.shape[1], U.shape[2]) )
    cluster_map = einops.rearrange(cluster_map, "Nx Ny -> (Nx Ny)")

    # fill in pixels we didn't use with Nones
    cluster_map[~sel] = None

    # the outliers will have index n_clusters
    cluster_map[ outlier_indices ] = n_clusters-1

    # place labels
    cluster_map[second_pass] = clustering.labels_

    # reshape
    cluster_map = einops.rearrange(cluster_map, "(Nx Ny) -> Nx Ny", Nx=U.shape[1], Ny=U.shape[2])
    return cluster_map