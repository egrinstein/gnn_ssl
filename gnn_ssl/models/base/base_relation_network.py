import torch
import torch.nn as nn

from .mlp import MLP
from .utils import ACTIVATIONS


# Base generic class to be inherited by the SSL specific network
class BaseRelationNetwork(nn.Module):
    def __init__(self, n_input_features, n_output_features,
                 n_pairwise_features,
                 on_input_hook=None, # None or nn.Module
                 on_pairwise_relation_network_start_hook=None, # None or nn.Module
                 pairwise_network_only=False,
                 activation="relu",
                 output_activation="sigmoid",
                 init_layers=True,
                 batch_norm=False,
                 dropout_rate=0,
                 n_layers=3,
                 signal_key=None):
        
        super().__init__()

        # 1. Store configuration
        self.n_input_features = n_input_features
        self.n_output_features = n_output_features
        self.n_pairwise_features = n_pairwise_features
        self.pairwise_network_only = pairwise_network_only
        self.n_input_relation_fusion_network = n_pairwise_features
        self.activation = ACTIVATIONS[activation]
        self.signal_key = signal_key

        # 2. Create local feature extractor, if provided
        self.on_input_hook = on_input_hook
        if on_input_hook is None:
            self.on_input_hook = DefaultOnInputHook(n_input_features)

        # 2. Create pairwise relation network
        self.pairwise_relation_network = PairwiseRelationNetwork(
            self.on_input_hook.n_output, self.n_pairwise_features,
            on_pairwise_relation_network_start_hook,
            activation, None, init_layers,
            batch_norm, dropout_rate, n_layers=n_layers,
            signal_key=signal_key, standalone=pairwise_network_only
        )
        
        # 3. Create relation fusion network
        self.relation_fusion_network = RelationFusionNetwork(
            self.n_input_relation_fusion_network, n_output_features,
            activation, output_activation, init_layers, batch_norm,
            dropout_rate, n_layers=n_layers, signal_key=signal_key
        )

    def forward(self, x):
        # x.shape == (batch_size, num_channels, feature_size)
        x = self.on_input_hook(x)
        # x.shape == (batch_size, num_channels, self.on_input_hook.n_output)
        x = self.pairwise_relation_network(x)
        # x.shape == (batch_size, self.n_pairwise_features)
        if self.pairwise_network_only:
            return x

        x = self.relation_fusion_network(x)
        # x.shape == (batch_size, self.n_output_features)
        return x


class PairwiseRelationNetwork(nn.Module):
    def __init__(self, n_input_features,
                       n_output_features,
                       on_start_hook=None,
                       activation="relu",
                       output_activation=None,
                       init_layers=True,
                       batch_norm=False,
                       dropout_rate=0,
                       n_layers=3,
                       signal_key=None,
                       standalone=False,
                       mask=True):

        super().__init__()

        self.signal_key = signal_key
        self.standalone = standalone
        self.output_activation = ACTIVATIONS[output_activation]
        self.mask = mask

        # 1. Create hook to be executed before network (Like extracting GCC-PHAT, or simply concatenating the pair)
        self.on_start_hook = on_start_hook
        if on_start_hook is None:
            self.on_start_hook = DefaultPairwiseRelationNetworkStartHook(n_input_features)
            # If pairwise feature extractor is provided, n_input_features refers to its output.
            # Else, it refers to the input of each object, therefore we double it.
        n_input_features = self.on_start_hook.n_output

        # 2. Create neural network
        self.pairwise_relation_network = MLP(n_input_features,
                                                n_output_features,
                                                n_output_features,
                                                activation,
                                                activation,
                                                batch_norm,
                                                dropout_rate,
                                                n_layers)

        if init_layers:
            for layer in self.pairwise_relation_network.layers:
                torch.nn.init.ones_(layer[0].weight)
            #init_layer(self.pairwise_relation_network)

    def forward(self, x):
        if self.signal_key is None:
            x_signal = x
        else:
            x_signal = x[self.signal_key]

        batch_size, n_channels, n_input_features = x_signal.shape

        # TODO: parallelize this?
        pairwise_relations = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                x_ij = self.on_start_hook(x, i, j)
                pairwise_relation = self.pairwise_relation_network(x_ij)#*x_ij[:, :625]
                # if self.mask:
                #     pairwise_relation *= x_ij[:, :pairwise_relation.shape[1]]
                pairwise_relations.append(pairwise_relation)
        
        pairwise_relations = torch.stack(pairwise_relations, dim=1)
        
        if self.standalone:
            x = pairwise_relations.mean(dim=1)
            if self.output_activation:
                x = self.output_activation(x)
            return x

        if self.signal_key is not None:
            x["signal"] = pairwise_relations
            return x
        else:
            pairwise_relations


class RelationFusionNetwork(nn.Module):
    def __init__(self, n_input_features,
                       n_output_features,
                       activation="relu",
                       output_activation="sigmoid",
                       init_layers=True,
                       batch_norm=False,
                       dropout_rate=0,
                       n_layers=3,
                       signal_key=None):

        super().__init__()

        self.relation_fusion_network = MLP(n_input_features,
                                           n_output_features,
                                           n_input_features,
                                           activation,
                                           output_activation,
                                           batch_norm,
                                           dropout_rate,
                                           n_layers)
        self.signal_key = signal_key

        if init_layers:
            for layer in self.relation_fusion_network.layers:
                torch.nn.init.eye_(layer[0].weight)
            #init_layer(self.relation_fusion_network)

    def forward(self, x):
        if self.signal_key is None:
            x_signal = x
        else:
            x_signal = x[self.signal_key]

        batch_size, n_channels, n_pairwise_features = x_signal.shape
        x_signal = x_signal.mean(dim=1)
        # x.shape == (batch_size, n_pairwise_features) 
        x_signal = self.relation_fusion_network(x_signal)
        # x.shape == (batch_size, self.n_output_features) 

        return x_signal


class DefaultPairwiseRelationNetworkStartHook(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()

        self.n_output = 2*n_input_features

    def forward(self, x, i, j):
        x_i = x[:, i]
        x_j = x[:, j]
        
        x_ij = torch.cat([x_i, x_j], axis=-1)
        
        return x_ij


class DefaultOnInputHook(nn.Module):
    """This is a placeholder for a torch.nn.Module which may be
    provider by a user who'd like for some preprocessing to occur on their input
    before sending it to the pairwise feature extractor.

    An example of an on_input_hook function may be a feature extractor such as the 
    Discrete Fourier transform to be applied individually to each channel.
    """
    def __init__(self, n_input_features):
        super().__init__()

        self.n_output = n_input_features

    def forward(self, x):
        return x