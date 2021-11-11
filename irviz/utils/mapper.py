import numpy as np
import einops


def selection_brackets_to_bool_array(selection_brackets, wavenumbers):
    """
    Converts a list of selected wavenumber brackets into a single boolean array

    Parameters
    ----------
    selection_brackets: List of selection brackets (some number, some numner)
    wavenumbers: an array of wavenumbers

    Returns
    -------
    Boolean array with selected wavenumbers

    """
    results = np.zeros(wavenumbers.shape[-1])
    for selector in selection_brackets:
        low_w = min(selector.values())
        high_w = max(selector.values())
        sel = (wavenumbers > low_w) & (wavenumbers < high_w)
        results[sel] = 1
    return results.astype(bool)


class einops_data_mapper(object):
    """
    This class allows one to map a tensor between (C X Y) and ( (X*Y) C ) forms.
    It takes into account absent data via a spatial mask (X,Y) and a spectral mask (C).
    """
    def __init__(self, tensor_shape, spatial_mask, spectral_mask):
        """

        Parameters
        ----------
        tensor_shape: The shape of the tensor that needs mapping from (i..e the IR data cube; (Nwav, Ny, Nx))
        spatial_mask: a boolean arra that indicates which pixels to use (Ny, Nx)
        spectral_mask: a boolean mask that indicates which wavenumbers to use (Nwav)
        """

        self.tensor_shape = tensor_shape

        self.spatial_mask = spatial_mask.astype(bool)
        self.N_obs = np.sum(self.spatial_mask)

        self.spectral_mask = spectral_mask.astype(bool)
        self.N_waves_used = np.sum(self.spectral_mask)

        self.spatial_mask_flat = einops.rearrange(self.spatial_mask, " Nx Ny -> (Nx Ny)")

    def spectral_tensor_to_spectral_matrix(self, spectral_map):
        """
        IR data cube to matrix for decomposition.

        Parameters
        ----------
        spectral_map: a numpy data set (Nwav,Ny,Nx)

        Returns
        -------
        A data matrix (Nx*Ny,Nwav)

        """
        data = einops.rearrange(spectral_map, " Nwav Nx Ny -> (Nx Ny) Nwav")
        data = data[:, self.spectral_mask]
        data = data[self.spatial_mask_flat]
        return data

    def matrix_to_tensor(self, data):
        """
        Map a (Nx*Ny, C) matrix to a (C,Ny,Nx) tensor

        Parameters
        ----------
        data: Input data

        Returns
        -------
        Output (C,Ny,Nx) tensor
        """

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
        """
        Map a (Nx*NY,Nwav) tensor to a (Nwav,Ny,Nx) tensor, taking into account missing data

        Parameters
        ----------
        data: input data (Nx2*Ny2,Nwav2), where Nx2*Ny2 comprises the numbner of active pixels in the spatial mask and
        Nwav2 the number of active wavenumbers.

        Returns
        -------
        A (Nwav,Ny,Nx) tensor with missing values as np.nan's
        """

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
        output_data[~fill_mask] = None
        output_data = einops.rearrange(output_data,
                                       "(X Y C) -> C X Y",
                                       C=self.tensor_shape[0],
                                       X=self.tensor_shape[1],
                                       Y=self.tensor_shape[2])
        return output_data
