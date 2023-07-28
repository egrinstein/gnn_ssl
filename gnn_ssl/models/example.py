import torch.nn as nn

from hydra.utils import get_class
from omegaconf import OmegaConf

from gnn_ssl.feature_extractors import get_stft_output_shape

from .base.mlp import MLP


class ExampleNet(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        feature_config, targets_config = config["features"], config["targets"]
        self.feature_extractor = get_class(
            config["features"]["class"])(config["features"])
        self.config = config = OmegaConf.to_object(config)

        n_frames, n_frame = get_stft_output_shape(feature_config)
        
        self.mlp = MLP(
            n_frame, config["n_output_features"], config["n_hidden_features"],
            config["n_layers"], config["activation"], config["output_activation"],
            config["batch_norm"], config["dropout_rate"]
        )

    def forward(self, x):
        x = {
            "signal": self.feature_extractor(x),
            "metadata": x["metadata"]
        }

        batch_size, n_mics, n_frames, n_frame = x["signal"].shape

        x = self.mlp(x["signal"])
        # x.shape == (batch_size, n_mics, n_frames, n_output_features)

        return {
            "example_output": x
        }
