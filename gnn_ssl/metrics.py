from turtle import forward
import torch

from torch.nn import Module

import pysoundloc.pysoundloc.metrics as ssl_metrics

from pysoundloc.pysoundloc.target_grids import (
    create_target_gaussian_grids, create_target_hyperbolic_grids
)


class Loss(Module):
    def __init__(self, config, dim=None):
        super().__init__()

        self.config = config

        self.loss = NormLoss(config["loss"], dim=dim)

        self.weighted = config["weighted_loss"]

        self.network_output_type = config["type"]
        if self.network_output_type == "grid":
            self.grid_generator = LikelihoodGrid(
                config["sigma"],
                config["n_points_per_axis"],
                config["hyperbolic_likelihood_weight"],
                config["gaussian_likelihood_weight"],
                normalize=config["normalize"],
                squared=config["squared"]
            )

    def forward(self, model_output, targets, mean_reduce=True):

        # 1. Prepare targets
        if self.network_output_type == "grid":
            targets = self.grid_generator(
                targets["room_dims"][..., :2], # :2 is because it's a 2-D grid
                targets["source_coordinates"][..., :2],
                targets["mic_coordinates"][..., :2],
            )
        elif self.network_output_type == "source_coordinates":
            targets = targets["source_coordinates"]

        model_output = model_output[self.network_output_type]

        # 2. Assert they have the same shape as output
        if model_output.shape != targets.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, targets.shape
            ))

        # Compute loss
        loss = self.loss(model_output, targets)
        if self.weighted:
            loss *= targets
        
        if mean_reduce:
            if self.weighted:
                loss = loss.sum()/loss.shape[0]
            else:
                loss = loss.mean()
        
        return loss
    

class LikelihoodGrid(Module):
    def __init__(self, sigma=1, n_points_per_axis=25,
                 hyperbolic_weight=0.5, gaussian_weight=0.5, normalize=True,
                 squared=False):
        super().__init__()

        self.sigma = sigma
        self.n_points_per_axis = n_points_per_axis
    
        self.hyperbolic_weight = hyperbolic_weight
        self.gaussian_weight = gaussian_weight
        self.normalize = normalize
        self.squared = squared

    def forward(self, room_dims, source_coordinates, mic_coordinates=None):
        batch_size = room_dims.shape[0]

        if mic_coordinates.shape[1] == 2:
            # Mixed grid is only used for two microphones
            gaussian_weight = self.gaussian_weight
            hyperbolic_weight = self.hyperbolic_weight
        else:
            # Compute gaussian grid only
            gaussian_weight = 1
            hyperbolic_weight = 0

        grids = []
        if gaussian_weight > 0:
            grids.append(
                create_target_gaussian_grids(
                        source_coordinates, room_dims,
                        self.n_points_per_axis,
                        self.sigma,
                        squared=self.squared)*gaussian_weight
            )
        if hyperbolic_weight > 0:
            if mic_coordinates is None:
                raise ValueError(
                    "mic_coordinates must be provided when computing hyperbolic grid")
            
            grids.append(
                create_target_hyperbolic_grids(
                    source_coordinates,
                    mic_coordinates[:, 0], mic_coordinates[:, 1],
                    room_dims, self.n_points_per_axis,
                    self.sigma,
                    squared=self.squared)*hyperbolic_weight
            )
        grids = torch.stack(grids)

        grids = grids.sum(dim=0)
        
        if self.normalize:
            max_grids = grids.abs().flatten(start_dim=1).max(dim=1)[0]
            min_grids = grids.abs().flatten(start_dim=1).min(dim=1)[0]

            max_grids = max_grids.unsqueeze(dim=1).unsqueeze(dim=2)
            min_grids = min_grids.unsqueeze(dim=1).unsqueeze(dim=2)
            
            # perform min-max normalization
            grids = (grids - min_grids)/(max_grids - min_grids)

        return grids


class NormLoss(Module):
    def __init__(self, norm_type="l1", dim=1, key=None):
        super().__init__()

        if norm_type not in ["l1", "l2", "squared"]:
            raise ValueError("Supported norms are 'l1', 'l2', and 'squared'")
        self.norm_type = norm_type
        self.dim = dim
        self.key = key

    def forward(self, model_output, targets, mean_reduce=False):
        # targets = targets.to(model_output.device)
        if self.key:
            model_output = model_output[self.key]
            targets = targets[self.key]
        error = model_output - targets
        if self.norm_type == "l1":
            normed_error = error.abs()
        elif self.norm_type == "l2" or self.norm_type == "squared":
            normed_error = error**2
        
        if self.dim is not None: # Sum along the vector dimension
            normed_error = torch.sum(normed_error, dim=self.dim)

        if self.norm_type == "l2":
            normed_error = torch.sqrt(normed_error)

        if mean_reduce:
            normed_error = normed_error.mean()

        return normed_error


class SourceDistance(NormLoss):
    def __init__(self):
        super().__init__("l2", 1, "source_coordinates")
    
    def forward(self, model_output, targets):
        if len(targets["source_coordinates"].shape) == 3: 
            targets["source_coordinates"] = targets["source_coordinates"][:, 0, :2]
        return super().forward(model_output, targets)


class SslMetrics(ssl_metrics.SslMetrics):
    def forward(self, model_output, targets):
        model_output = model_output["source_coordinates"]
        room_dims = targets["room_dims"]
        targets = targets["source_coordinates"][:, :2]
        return super().forward(model_output, targets, room_dims)
