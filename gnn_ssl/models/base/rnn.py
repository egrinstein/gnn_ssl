import math
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, n_input, n_hidden,
                 bidirectional=False, output_mode="last_time", n_layers=1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.bidirectional = bidirectional
        self.output_mode = output_mode

        if bidirectional:
            n_hidden //=2

        super().__init__()

        self.rnn = nn.GRU(input_size=n_input,
                          hidden_size=n_hidden,
                          batch_first=True,
                          bidirectional=bidirectional,
                          num_layers=n_layers)

    def forward(self, x):
        (x, _) = self.rnn(x)
        # (batch_size, time_steps, feature_maps):
        # Select last time step
        if self.output_mode == "last_time":
            return x[:, -1]
        elif self.output_mode == "avg_time":
            return x.mean(dim=1)
        elif self.output_mode == "avg_channels":
            return x.mean(dim=2)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)
