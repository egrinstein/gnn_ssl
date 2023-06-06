import torch.nn as nn

from pysoundloc.pysoundloc.models import srp_phat

class SrpPhat(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        self.feature_config, self.targets_config = config["features"], config["targets"]
        self.n_points_per_axis = self.targets_config["n_points_per_axis"]
        self.sr = config["features"]["dataset"]["sr"]
        self.thickness = config["features"]["srp_thickness"]

    def forward(self, x):
        mic_signals = x["signal"]
        mic_coords = x["metadata"]["local"]["mic_coordinates"][..., :2]
        room_dims = x["metadata"]["global"]["room_dims"][..., :2]
        return srp_phat(mic_signals, mic_coords, room_dims,
                        self.sr, self.n_points_per_axis,
                        thickness=self.thickness)
