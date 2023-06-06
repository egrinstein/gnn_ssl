import torch
import torch.nn as nn

from .mlp import fc_block
from .utils import ACTIVATIONS, AvgLayer, init_layer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1),
                padding=(1,1), dilation=(1, 1),
                pool_size=(2, 2), pool_type="avg",
                block_type="double",
                dropout_rate=0, batch_norm=False):
        
        super().__init__()

        # Dump parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool_size = pool_size
        self.block_type = block_type
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        blocks = [
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
        ]
        
        if batch_norm:
            blocks.append(nn.BatchNorm2d(out_channels))

        blocks.append(nn.ReLU())

        if block_type == "double": 
            blocks.append(nn.Conv2d(in_channels=out_channels, 
                                   out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation))
            if batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU())
        

        # Create activation, dropout and pooling blocks
        if pool_type == "avg":
            blocks.append(nn.AvgPool2d(pool_size))
        elif pool_type == "max":
            blocks.append(nn.MaxPool2d(pool_size))
        else:
            raise ValueError(f"pool_type '{pool_type}' is invalid. Must be 'max' or 'avg'")

        if dropout_rate > 0:
            blocks.append(nn.Dropout(dropout_rate))

        self.model = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.model(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def get_output_shape(self, input_shape):
        """
        Args:
            input_shape (tuple): (in_width, in_height)

        Returns:
            tuple: (out_width, out_height)
        """

        output_shape = get_conv2d_output_shape(
            input_shape, self.kernel_size, self.stride, self.dilation, self.padding)

        if self.block_type == "double":
            output_shape = get_conv2d_output_shape(
                output_shape, self.kernel_size, self.stride, self.dilation, self.padding)
        output_shape = get_pool2d_output_shape(output_shape, self.pool_size)

        return output_shape


class Cnn1d(nn.Module):
    def __init__(self, n_input_channels, n_output_features, n_hidden_channels,
                 activation, output_activation, batch_norm, n_layers, n_metadata):
        super().__init__()

        activation = ACTIVATIONS[activation]
        output_activation = ACTIVATIONS[output_activation]

        layers = [
            _conv_1d_block(n_input_channels, n_hidden_channels, activation, batch_norm=batch_norm)
        ]

        for _ in range(n_layers - 1): # -1 = skipping input layer
            layers.append(
                _conv_1d_block(n_hidden_channels,
                            n_hidden_channels,
                            activation,
                            batch_norm=batch_norm
                )
            )

        layers.append(AvgLayer(dim=2)) # dim0 = batch, dim1 = channels, dim2 = time_steps

        self.conv_layers = nn.Sequential(*layers)
        self.fc_layer = fc_block(
            n_hidden_channels + 2*n_metadata, n_output_features,
            activation=output_activation, batch_norm=False
        )
        self.n_metadata = n_metadata
    
    def forward(self, x):
        batch_size, n_channels, n_time_steps = x.shape

        if self.n_metadata > 0:
            x = x[:, :, :-self.n_metadata]
            metadata = x[:, :, -self.n_metadata:]
            metadata = metadata.reshape((batch_size, n_channels*self.n_metadata))
        
        x = self.conv_layers(x)
        
        if self.n_metadata > 0:
            x = torch.cat([x, metadata], dim=1)

        x = self.fc_layer(x)

        return x


def _conv_1d_block(n_input_channels, n_output_channels,
                   activation, kernel_size=3, stride=1,
                   padding=1, pool_size=1, pool_type="avg",
                   batch_norm=False):
        
    layers = [
        nn.Conv1d(n_input_channels, n_output_channels,
                    kernel_size, stride, padding)
    ]
    
    if activation:
        layers.append(activation())
    if batch_norm:
        layers.append(nn.BatchNorm1d(n_output_channels))
    if pool_size > 1:
        if pool_type == "avg":
            layers.append(nn.AvgPool1d(pool_size))
        elif pool_type == "max":
            layers.append(nn.MaxPool1d(pool_size))
        else:
            raise ValueError("Pooling layer must be 'max' or 'avg'")
    
    return nn.Sequential(*layers)


def get_conv2d_output_shape(input_shape: tuple,
                            kernel_size: tuple,
                            stride=(1, 1),
                            dilation=(1, 1),
                            padding=(0, 0)):
    """Compute the output of a convolutional layer.
    See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    for more information.
    """

    input_shape = torch.Tensor(input_shape)
    kernel_size = torch.Tensor(kernel_size)
    stride = torch.Tensor(stride)
    dilation = torch.Tensor(dilation)
    padding = torch.Tensor(padding)

    n_output_shape = (input_shape + 2*padding - kernel_size - (dilation - 1)*(kernel_size - 1))/stride + 1
    #print(n_output_shape)
    n_output_shape = torch.floor(n_output_shape)

    return n_output_shape


def get_pool2d_output_shape(n_input_shape, pool_size):
    return get_conv2d_output_shape(n_input_shape,
                                   pool_size,
                                   pool_size)


def get_conv_pool_output_shape(n_input_shape: tuple,
                               kernel_size: tuple,
                               pool_size: tuple,
                               stride=(1, 1),
                               dilation=(1, 1),
                               padding=(0, 0)):

    conv_output = get_conv2d_output_shape(
        n_input_shape, kernel_size, stride,
        dilation, padding
    )
    pool_output = get_pool2d_output_shape(conv_output, pool_size)

    return pool_output
