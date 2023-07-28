from hydra.utils import get_class
from omegaconf import OmegaConf

from gnn_ssl.feature_extractors.pairwise_feature_extractors import ArrayWiseSpatialLikelihoodGrid
from gnn_ssl.feature_extractors.pairwise_feature_extractors import (
    GccPhat, MetadataAwarePairwiseFeatureExtractor, SpatialLikelihoodGrid
)

from pysoundloc.pysoundloc.utils.math import grid_argmax

from .base.mlp import MLP
from .base.base_relation_network import BaseRelationNetwork

SIGNAL_KEY = "signal"


class GnnSslNet(BaseRelationNetwork):
    def __init__(self,
                 config,
                 **kwargs):

        self.config = config = OmegaConf.to_object(config)

        feature_config, target_config = config["features"], config["targets"]
        dataset_config = feature_config["dataset"]

        # 1. Store configuration
        self.is_metadata_aware = config["is_metadata_aware"]
        self.use_rt60_as_metadata = dataset_config["use_rt60_as_metadata"]
        self.n_likelihood_grid_points_per_axis = target_config["n_points_per_axis"]
        self.sr = dataset_config["sr"]
        n_input_features = int(dataset_config["n_input_seconds"]*self.sr)
        # Set output size
        self.output_target = target_config["type"]
        if self.output_target == "source_coordinates":
            n_output_features = 2 # x, y coords of the microphones
        else:
            n_output_features = self.n_likelihood_grid_points_per_axis**2

        # 2. Create local feature extractor
        if config["local_feature_extractor"] and config["pairwise_feature_extractor"]:
            raise ValueError(
                """Simultaneously using local and pairwise
                feature extractors is not yet supported.""")

        if config["local_feature_extractor"] == "mlp":
            config["local_feature_extractor"] = MLP(n_input_features,
                                          config["n_pairwise_features"],
                                          config["n_pairwise_features"],
                                          config["activation"],
                                          None,
                                          config["batch_norm"],
                                          config["dropout_rate"],
                                          config["n_layers"])
        elif config["local_feature_extractor"] == "slf":
            config["local_feature_extractor"] = ArrayWiseSpatialLikelihoodGrid(
                self.sr, self.n_likelihood_grid_points_per_axis,
                thickness=feature_config["srp_thickness"]
            )
        if config["local_feature_extractor"] is not None:
            n_input_features = config["local_feature_extractor"].n_output


        super().__init__(n_input_features, n_output_features, config["n_pairwise_features"], config["local_feature_extractor"],
                    config["pairwise_feature_extractor"], config["pairwise_network_only"],
                    config["activation"], config["output_activation"], config["init_layers"], config["batch_norm"],
                    config["dropout_rate"], config["n_layers"], SIGNAL_KEY)

        # 3. Create pairwise feature extractor
        self.use_pairwise_feature_extractor = config["pairwise_feature_extractor"] is not None
        if self.use_pairwise_feature_extractor:
            if config["pairwise_feature_extractor"] == "gcc_phat":
                config["pairwise_feature_extractor"] = GccPhat(self.sr, feature_config["n_dft"])
            elif config["pairwise_feature_extractor"] == "spatial_likelihood_grid":
                config["pairwise_feature_extractor"] = SpatialLikelihoodGrid(self.sr, self.n_likelihood_grid_points_per_axis)
            else:
                raise ValueError("pairwise_feature_extractor must be 'gcc_phat' or 'spatial_likelihood_grid'")

            n_input_features = pairwise_feature_extractor.n_output
    
        pairwise_feature_extractor = MetadataAwarePairwiseFeatureExtractor(
            n_input_features,
            config["pairwise_feature_extractor"],
            self.is_metadata_aware,
            self.use_rt60_as_metadata
        )

    def forward(self, x, estimate_coords=False):
        y = super().forward(x)

        if estimate_coords and self.output_target == "likelihood_grid":
            batch_size = y.shape[0]
            estimated_coords = grid_argmax(
                y.reshape(batch_size, self.n_likelihood_grid_points_per_axis, self.n_likelihood_grid_points_per_axis),
                x["global"]["room_dims"])

            return estimated_coords, y
        else:
            return y
