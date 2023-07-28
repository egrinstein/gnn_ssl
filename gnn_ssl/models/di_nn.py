import torch
import torch.nn as nn

from hydra.utils import get_class
from omegaconf import OmegaConf

from gnn_ssl.feature_extractors import get_stft_output_shape
from pysoundloc.pysoundloc.utils.math import grid_argmax

from .base.rnn import RNN, init_gru
from .base.mlp import MLP
from .base.cnn import ConvBlock
from ..feature_extractors.metadata import flatten_metadata


class DINN(nn.Module):
    """
    CRNN model that accepts a secondary input
    consisting of metadata (microphone positions, room dimensions, rt60)
    """
    def __init__(self, config):
        
        super().__init__()

        feature_config = config["features"]
        dataset_config = feature_config["dataset"]
        targets_config = config["targets"]


        self.feature_extractor = get_class(
            config["features"]["class"])(config["features"])
        self.config = config = OmegaConf.to_object(config)

        # 1. Store configuration
        self.n_input_channels = config["n_mics"]
        self.pool_type = config["pool_type"]
        self.pool_size = config["pool_size"]
        self.kernel_size = config["kernel_size"]

        self.input_shape = get_stft_output_shape(feature_config)

        self.is_metadata_aware = config["is_metadata_aware"]
        self.metadata_dim = dataset_config["metadata_dim"]

        self.grid_size = targets_config["n_points_per_axis"]
        self.output_type = targets_config["type"] # grid or regression vector
        if self.output_type == "source_coordinates":
            config["encoder_mlp_config"]["n_output_channels"] = targets_config["n_output_coordinates"]

        # 3. Create encoder
        self.encoder = Encoder(self.n_input_channels,
                               self.input_shape,
                               config["conv_layers_config"],
                               config["rnn_config"],
                               config["encoder_mlp_config"],
                               config["init_layers"],
                               config["pool_type"],
                               config["pool_size"],
                               config["kernel_size"],
                               flatten_dims=config["flatten_dims"],
                               batch_norm=config["batch_norm"],
                               is_metadata_aware=self.is_metadata_aware,
                               metadata_dim=dataset_config["metadata_dim"])

        if config["output_activation"] == "relu":
            self.output_activation = nn.ReLU()
        elif config["output_activation"] == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = None

    def forward(self, x):
        room_dims = x["metadata"]["global"]["room_dims"][:, :2]

        x = {
            "signal": self.feature_extractor(x),
            "metadata": x["metadata"]
        }        

        x = self.encoder(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        if self.output_type == "grid":
            batch_size = x.shape[0]
            x = x.reshape((batch_size, self.grid_size, self.grid_size))
            return {
                "source_coordinates": grid_argmax(x, room_dims),
                "grid": x
            }
        else:
            return {
                "source_coordinates": x
            }


class Encoder(nn.Module):
    def __init__(self, n_input_channels,
                       input_shape,
                       conv_layers_config,
                       rnn_config,
                       mlp_config,
                       init_layers,
                       pool_type,
                       pool_size,
                       kernel_size,
                       flatten_dims=False,
                       batch_norm=False,
                       is_metadata_aware=True,
                       metadata_dim=3):
    
        super().__init__()

        # 1. Store configuration
        self.n_input_channels = n_input_channels
        self.input_shape = input_shape
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.n_conv_output_output = conv_layers_config[-1]["n_channels"]
        self.n_rnn_output_channels = rnn_config["n_output_channels"]
        self.flatten_dims = flatten_dims
        self.batch_norm = batch_norm
        self.metadata_dim = metadata_dim # 3 for 3D localization, 2 for 2 for 2D localization
        self.is_metadata_aware = is_metadata_aware

        # 2. Create convolutional blocks
        self.conv_blocks, self.conv_output_shape = self._create_conv_blocks(
            conv_layers_config, batch_norm=batch_norm
        )

        # 3. Create recurrent block
        self.n_rnn_input = self.n_conv_output_output
        if flatten_dims:
            # The input for the RNN will be
            # The frequency_bins x conv_output_channels
            self.n_rnn_input = int(self.n_rnn_input*self.conv_output_shape[-1])

        self.rnn = RNN(self.n_rnn_input,
                       self.n_rnn_output_channels,
                       rnn_config["bidirectional"],
                       rnn_config["output_mode"],
                       n_layers=rnn_config["n_layers"]
        )

        # 4. Create mlp block
        n_input_mlp = self.n_rnn_output_channels
        if is_metadata_aware:
            n_metadata = metadata_dim*(self.n_input_channels + 1) # 1 => room dims 
            n_input_mlp += n_metadata

        self.mlp = MLP(n_input_mlp, mlp_config["n_output_channels"],
                        n_hidden_features=mlp_config["n_hidden_channels"],
                        n_layers=mlp_config["n_layers"],
                        batch_norm=False, # Batch normalization on the MLP slowed training down.
                        dropout_rate=mlp_config["dropout_rate"]) 
        if init_layers:
            init_gru(self.rnn.rnn)

    def forward(self, x):

        if self.is_metadata_aware:
            metadata = flatten_metadata(x["metadata"])

        x = x["signal"]
        # (batch_size, num_channels, time_steps, freqs)

        # 1. Extract features using convolutional layers
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # (batch_size, feature_maps, time_steps', freqs')

        # 2. Average across all frequency bins,
        # or flatten the frequency and channels

        if self.flatten_dims:
            x = x.transpose(2, 3)
            x = x.flatten(start_dim=1, end_dim=2)
        else:
            if self.pool_type == "avg":
                x = torch.mean(x, dim=3)
            elif self.pool_type == "max":
                x = torch.max(x, dim=3)[0]

        # (batch_size, feature_maps, time_steps)

        # Preprocessing for RNN
        x = x.transpose(1,2)
        # (batch_size, time_steps, feature_maps):

        # 3. Apply RNN
        x = self.rnn(x)


        if self.is_metadata_aware:
            # Concatenate metadata before sending to fully connected layer,
            x = torch.cat([x, metadata], dim=1)
            # (batch_size, n_metadata_unaware_features + n_metadata)
        
        # 5. Fully connected layer
        x = self.mlp(x)
        # (batch_size, class_num)

        return x

    def _create_conv_blocks(self, conv_layers_config, batch_norm):
        conv_blocks = [
            ConvBlock(self.n_input_channels, conv_layers_config[0]["n_channels"],
                      block_type=conv_layers_config[0]["type"],
                      pool_size=self.pool_size,
                      pool_type=self.pool_type,
                      kernel_size=self.kernel_size,
                      batch_norm=batch_norm)
        ]
        current_output_shape = conv_blocks[-1].get_output_shape(self.input_shape)

        for i, config in enumerate(conv_layers_config[1:]):
            last_layer = conv_blocks[-1]
            in_channels = last_layer.out_channels
            conv_blocks.append(
                ConvBlock(in_channels, config["n_channels"],
                          block_type=config["type"],
                          pool_size=self.pool_size,
                          pool_type=self.pool_type,
                          kernel_size=self.kernel_size,
                          batch_norm=batch_norm)
            )
            current_output_shape = conv_blocks[-1].get_output_shape(current_output_shape)
        return nn.ModuleList(conv_blocks), current_output_shape
