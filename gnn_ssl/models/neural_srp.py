import torch

from pysoundloc.pysoundloc.math_utils import grid_argmax

from .pairwise_neural_srp import PairwiseNeuralSrp
from ..feature_extractors.metadata import filter_local_metadata


class NeuralSrp(PairwiseNeuralSrp):
    """Multi-microphone version of PairwiseNeuralSrp,
    which works for a single pair of microphones.

    NeuralSrp computes a PairwiseNeuralSrp grid for each microphone pair,
    then sums them together.

    """

    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, x, estimate_coords=True, mean=True):
        # x = self.feature_extractor(x) TODO: Extract features before for efficiency
        # batch_size, n_channels, n_time_bins, n_freq_bins = x["signal"].shape
        batch_size, n_channels, n_time_samples = x["signal"].shape

        room_dims = x["metadata"]["global"]["room_dims"][:, :2]

        y = []
        n_pairs = n_channels*(n_channels - 1)/2
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                x_ij = {
                    "signal": x["signal"][:, [i, j]],
                    "metadata": filter_local_metadata(x["metadata"], [i, j])
                }
                
                y.append(super().forward(x_ij)["grid"])
                #count += 1

        y = torch.stack(y, dim=1).sum(dim=1)

        if mean:
            y /= n_pairs

        if estimate_coords:
            estimated_coords = grid_argmax(y, room_dims)

            return {
                "source_coordinates": estimated_coords,
                "grid": y    
            }

        return y
