import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "sigmoid": nn.Sigmoid,
    None: None
}


class MLP(nn.Module):
    def __init__(self, n_input_features, n_output_features,
                n_hidden_features, n_layers, activation="relu",
                output_activation=None, batch_norm=False, dropout_rate=0):

        super().__init__()

        activation = ACTIVATIONS[activation]
        output_activation = ACTIVATIONS[output_activation]

        layers = [
            fc_block(n_input_features, n_hidden_features, activation, batch_norm, dropout_rate)
        ]

        for _ in range(n_layers - 2): # -2 = skipping input and output layers
            layers.append(
                fc_block(n_hidden_features,
                          n_hidden_features,
                          activation,
                          batch_norm=batch_norm
                )
            )

        layers.append(fc_block(
            n_hidden_features, n_output_features,
            activation=output_activation, batch_norm=False)
        )

        self.layers = nn.Sequential(*layers)
        self.n_output = n_output_features

    def forward(self, x):
        return self.layers(x)


def fc_block(n_input, n_output, activation, batch_norm=False, dropout_rate=0):
    layers = [nn.Linear(n_input, n_output)]
    if activation:
        layers.append(activation())
    if batch_norm:
        layers.append(nn.BatchNorm1d(n_output))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*layers)
