import math
import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "sigmoid": nn.Sigmoid,
    None: None
}


class MaxLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.max(dim=self.dim)


class AvgLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.mean(dim=self.dim)


def init_layer(layer, nonlinearity='relu'):
    """Initialize a convolutional or linear layer
    Credits to Yin Cao et al:
    https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization/blob/master/models/model_utilities.py
    """

    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        #nn.init.normal_(layer.weight, 1.0, 0.02)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


def load_checkpoint(model, checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = {}

    for k, v in checkpoint["state_dict"].items():
        k = _remove_prefix(k, "model.")
        state_dict[k] = v

    model.load_state_dict(state_dict)


def _remove_prefix(s, prefix):
    return s[len(prefix):] if s.startswith(prefix) else s
