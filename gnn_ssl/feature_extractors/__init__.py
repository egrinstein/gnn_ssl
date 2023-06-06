import math
import torch
import torch.nn as nn

from omegaconf import OmegaConf

from .stft import CrossSpectra, StftArray


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = OmegaConf.to_object(config)

        self.n_input_seconds = config["dataset"]["n_input_seconds"]
        self.config = config

        feature_name = config["type"]

        self.n_output_channels = 2
        if feature_name.startswith("stft"):
            if feature_name == "stft":
                self.n_output_channels = 4 # 2 mics, each with a real and imaginary channel
                mag_only = phase_only = real_only = False
            elif feature_name == "stft_phase":
                mag_only = real_only = False
                phase_only = True
            elif feature_name == "stft_mag":
                phase_only = real_only = False
                mag_only = True
            elif feature_name == "stft_real":
                phase_only = mag_only = False
                real_only = True
            else:
                raise ValueError(f"{feature_name} is not a valid feature extractor")
            self.model = StftArray(
                is_complex=False, complex_as_channels=True,
                real_only=real_only, mag_only=mag_only, phase_only=phase_only,
                n_dft=config["n_dft"], hop_size=config["hop_size"]
            )
        elif feature_name == "cross_spectral_phase":
            self.model = CrossSpectra(phase_only=True,
                            n_dft=config["n_dft"], hop_size=config["hop_size"])
            self.n_output_channels = 1
    
        self.n_output = self._get_output_shape()

    def forward(self, x):
        return self.model(x["signal"])

    def _get_output_shape(self):
        n_input_samples = self.n_input_seconds*self.config["dataset"]["sr"]

        out_width = math.ceil((n_input_samples)/self.config["hop_size"])
        out_height = self.config["n_dft"]//2 # /2 as we use "onesided" dft
        
        return (out_width, out_height)


def get_stft_output_shape(feature_config):
    n_input_samples = feature_config["dataset"]["n_input_seconds"]*feature_config["dataset"]["sr"]

    out_width = math.ceil((n_input_samples)/feature_config["hop_size"])
    out_height = feature_config["n_dft"]//2 # /2 as we use "onesided" dft
    
    return torch.Tensor((out_width, out_height))
