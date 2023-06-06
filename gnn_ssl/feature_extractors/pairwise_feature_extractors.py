import torch

from torch.nn import Module

from pysoundloc.pysoundloc.correlation import gcc_phat_batch
from pysoundloc.pysoundloc.srp import compute_pairwise_srp_grids


# Every pairwise feature extractor inherits from the torch.nn.Module class
# and has a n_output variable.
# Furthermore, it implements a "forward" function with three parameters:
# A tensor x and two indices i and j


class GccPhat(Module):
    def __init__(self, sr, n_bins, normalize=True):
        self.sr = sr
        self.n_output = n_bins
        self.normalize = normalize
        super().__init__()
    
    def forward(self, x, i, j):
        x_i = x["signal"][:, i]
        x_j = x["signal"][:, j]

        batch_size, n_signal = x_i.shape
        results = []

        results = gcc_phat_batch(x_i, x_j, self.sr)[0]
        # Only select central self.n_output
        results = results[:, (n_signal - self.n_output)//2:(n_signal + self.n_output)//2]

        if self.normalize:
            results /= results.max(dim=1, keepdims=True)[0]
        return results


class SpatialLikelihoodGrid(Module):
    def __init__(self, sr, n_grid_points, flatten=True, normalize=False):
        super().__init__()
        self.sr = sr
        self.n_grid_points = n_grid_points

        self.flatten = flatten
        self.normalize = normalize

        if flatten:
            self.n_output = n_grid_points**2
        else:
            self.n_output = (n_grid_points, n_grid_points)
    
    def forward(self, x, i, j):
        x_i = x["signal"][:, i]
        x_j = x["signal"][:, j]

        mic_i_coords = x["local"]["mic_coordinates"][:, i]
        mic_j_coords = x["local"]["mic_coordinates"][:, j]
        room_dims = x["global"]["room_dims"]

        grids = compute_pairwise_srp_grids(
            x_i, x_j, self.sr,
            mic_i_coords, mic_j_coords,
            room_dims,
            n_grid_points=self.n_grid_points
        )

        if self.flatten:
            grids = grids.flatten(start_dim=1)
            if self.normalize:
                grids /= grids.max(dim=-1)[0].unsqueeze(1)

        return grids

   
class MetadataAwarePairwiseFeatureExtractor(Module):
    def __init__(self, n_input, pairwise_feature_extractor=None,
                 is_metadata_aware=True, use_rt60_as_metadata=True):
        super().__init__()

        self.pairwise_feature_extractor = pairwise_feature_extractor
        
        if pairwise_feature_extractor is None:
            self.n_output = 2*n_input # 2 channels will be concatenated
        else:
            self.n_output = pairwise_feature_extractor.n_output

        self.is_metadata_aware = is_metadata_aware
        self.use_rt60_as_metadata = use_rt60_as_metadata

        if is_metadata_aware:
            self.n_output += 2*2*2 + 2 # 2 coordinates times 2 microphones per array + 2 room_dims

            if use_rt60_as_metadata:
                self.n_output += 1

    def forward(self, x, i, j):
        # 1. Apply pairwise feature extractor, if provided. Else, simply concat the signals
        if self.pairwise_feature_extractor is not None:
            x_ij = self.pairwise_feature_extractor(x, i, j)
        else:
            x_ij = torch.cat([x["signal"][:, i], x["signal"][:, j]], dim=1)

        if not self.is_metadata_aware:
            return x_ij
        
        # 2. Concatenate local metadata
        mic_coords = x["local"]["mic_coordinates"]
        room_dims = x["global"]["room_dims"]
        x_ij = torch.cat([
            x_ij, mic_coords[:, i].flatten(start_dim=1), mic_coords[:, j].flatten(start_dim=1),
            room_dims], dim=1)

        if not self.use_rt60_as_metadata:
            return x_ij

        # 3. Concatenate global_metadata        
        rt60 = x["global"]["rt60"]
        x_ij = torch.cat([x_ij, room_dims, rt60], dim=1)

        return x_ij


class ArrayWiseSpatialLikelihoodGrid(Module):
    """
    Compute the cumulative Spatial Likelihood Function (SLF)
    using the signals of a microphone array, their respective microphone coordinates
    and the room dimensions.

    If microphone_wise == True, one cumulative SLF will be produced for each microphone,
    using the correlations between it and the other ones.
    """

    def __init__(self, sr, n_grid_points, flatten=True, thickness=10):
        super().__init__()
        self.sr = sr
        self.n_grid_points = n_grid_points
        self.thickness=thickness

        self.flatten = flatten
        if flatten:
            self.n_output = n_grid_points**2
        else:
            self.n_output = (n_grid_points, n_grid_points)

    def forward(self, x):
        x_signal = x["signal"]
        mic_coords = x["local"]["mic_coordinates"]
        room_dims = x["global"]["room_dims"]

        batch_size, n_arrays, n_array, n_signal = x_signal.shape

        grids = torch.zeros((
            batch_size, n_arrays,
            self.n_grid_points, self.n_grid_points
        ))

        for k in range(n_arrays):
            for i in range(n_array):
                for j in range(i + 1, n_array):
                    grid_ij = compute_pairwise_srp_grids(
                            x_signal[:, k, i], x_signal[:, k, j], self.sr,
                            mic_coords[:, k, i], mic_coords[:, k, j],
                            room_dims, n_grid_points=self.n_grid_points,
                            thickness=self.thickness
                    )
                    grids[:, k] += grid_ij

        if self.flatten:
            grids = grids.flatten(start_dim=2)
            
        # Normalize grids
        # Only works on flattened!
        max_grids = grids.max(dim=2)[0].unsqueeze(2)
        grids /= max_grids

        x["signal"] = grids
        return x
