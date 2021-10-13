import numpy as np
from sklearn.cluster import KMeans
from irviz.background_app import open_map_file
from irviz.decomposition_methods.simple_PCA import masked_to_map, simple_PCA

TEST_FILE = 'E:\\BP-area3a.h5'


def kmeansClustering(data_transform, mask, n_clusters=5, random_state=0):
    """
    K-means clustering
    Parameters
    ----------
    mask: binary 2D mask of shape [Ny, Nx], numpy array
    data_transform: Transformed spectral map of shape[Nw, Ny, Nx], dask array
    n_clusters: number of clusters
    random_state: random state number

    Returns
    -------
    label_map: A cluster map of size [Ny, Nx], may contain np.NANs
    """
    data_transform = np.array(data_transform)
    data_transform = data_transform.transpose((1, 2, 0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data_transform[mask])
    labels = kmeans.labels_
    label_map = masked_to_map(mask, labels)
    return label_map

if __name__ == "__main__":
    data, bounds = open_map_file(TEST_FILE)
    wavenumbers = bounds[0]
    mask = np.random.random(data.shape[1:3]) > 0.5
    control_regions = [{'region_min': 1200, 'region_max': 1400}, {'region_min': 2700, 'region_max': 3000}]

    data_transform, vec = simple_PCA(wavenumbers, data, mask, control_regions)
    label_map = kmeansClustering(mask, data_transform)
    assert label_map.shape == (29, 42), "shape of label map is wrong."