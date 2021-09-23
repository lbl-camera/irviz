import numpy as np
import einops

class einops_data_mapper(object):
    """
    This class allows one to map a tensor between (C X Y) and ( (X*Y) C ) forms.
    It takes into account absent data via a spatial mask (X,Y) and a spectral mask (C).
    """
    def __init__(self, tensor_shape, spatial_mask, spectral_mask):

        self.tensor_shape = tensor_shape

        self.spatial_mask = spatial_mask.astype(bool)
        self.N_obs = np.sum(self.spatial_mask)

        self.spectral_mask = spectral_mask.astype(bool)
        self.N_waves_used = np.sum(self.spectral_mask)

        self.spatial_mask_flat = einops.rearrange(self.spatial_mask, " Nx Ny -> (Nx Ny)")

    def spectral_tensor_to_spectral_matrix(self, spectral_map):
        data = einops.rearrange(spectral_map, " Nwav Nx Ny -> (Nx Ny) Nwav")
        data = data[:, self.spectral_mask]
        data = data[self.spatial_mask_flat]
        return data

    def matrix_to_tensor(self, data):
        N_channels = data.shape[1]
        N_obs = data.shape[0]
        assert N_obs == self.N_obs

        # build output tensor
        output_data = np.zeros( (self.tensor_shape[1]*self.tensor_shape[2],N_channels) )

        # take care of missing data
        output_data[~self.spatial_mask_flat, :] = None
        # fill in the rest
        output_data[self.spatial_mask_flat, :] = data
        # rearrange things
        output_data = einops.rearrange(output_data,
                                       "(X Y) C -> C X Y",
                                       C=N_channels,
                                       X=self.tensor_shape[1],
                                       Y=self.tensor_shape[2])
        return output_data

    def spectral_matrix_to_spectral_tensor(self, data):
        N_channels = data.shape[1]
        N_obs = data.shape[0]
        assert N_obs == self.N_obs
        assert N_channels == self.N_waves_used

        output_data = np.zeros(self.tensor_shape)
        output_data = einops.rearrange(output_data, "Nwav Nx Ny -> (Nx Ny Nwav)")

        fill_mask = np.ones(self.tensor_shape)
        fill_mask = einops.rearrange(fill_mask, "Nwav Nx Ny -> (Nx Ny) Nwav")
        fill_mask[ ~self.spatial_mask_flat, : ] = 0
        fill_mask[:, ~self.spectral_mask] = 0
        fill_mask = einops.rearrange(fill_mask, "A C -> (A C)")
        fill_mask = fill_mask.astype(bool)

        output_data[ fill_mask ] = einops.rearrange(data, "A C -> (A C)")
        output_data = einops.rearrange(output_data,
                                       "(X Y C) -> C X Y",
                                       C=self.tensor_shape[0],
                                       X=self.tensor_shape[1],
                                       Y=self.tensor_shape[2])
        return output_data
