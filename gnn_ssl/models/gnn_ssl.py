from gnn_ssl.feature_extractors.pairwise_feature_extractors import ArrayWiseSpatialLikelihoodGrid
from gnn_ssl.feature_extractors.pairwise_feature_extractors import (
    GccPhat, MetadataAwarePairwiseFeatureExtractor, SpatialLikelihoodGrid
)

from pysoundloc.pysoundloc.math_utils import grid_argmax

from .base.utils import MLP
from .base_relation_network import BaseRelationNetwork

SIGNAL_KEY = "signal"


class GnnSslNet(BaseRelationNetwork):
    def __init__(self,
                 n_input_seconds,
                 n_pairwise_features,
                 target_config=None,
                 feature_extraction_config=None, # TODO: Remove feature extractor
                 pairwise_feature_extractor=None,
                 pairwise_network_only=False,
                 local_feature_extractor=None,
                 activation="relu",
                 output_activation="sigmoid",
                 init_layers=True,
                 is_metadata_aware=True,
                 use_rt60_as_metadata=True,
                 batch_norm=False,
                 n_layers=3,
                 dropout_rate=0,
                 **kwargs):

        # 1. Store configuration
        self.is_metadata_aware = is_metadata_aware
        self.use_rt60_as_metadata = use_rt60_as_metadata
        self.n_likelihood_grid_points_per_axis = target_config["n_points_per_axis"]
        self.sr = feature_extraction_config["dataset"]["sr"]
        n_input_features = int(n_input_seconds*self.sr)
        # Set output size
        self.output_target = target_config["type"]
        if self.output_target == "source_coordinates":
            n_output_features = 2 # x, y coords of the microphones
        else:
            n_output_features = self.n_likelihood_grid_points_per_axis**2

        # 2. Create local feature extractor
        if local_feature_extractor and pairwise_feature_extractor:
            raise ValueError(
                """Simultaneously using local and pairwise
                feature extractors is not yet supported.""")

        if local_feature_extractor == "mlp":
            local_feature_extractor = MLP(n_input_features,
                                          n_pairwise_features,
                                          n_pairwise_features,
                                          activation,
                                          None,
                                          batch_norm,
                                          dropout_rate,
                                          n_layers)
        elif local_feature_extractor == "slf":
            local_feature_extractor = ArrayWiseSpatialLikelihoodGrid(
                self.sr, self.n_likelihood_grid_points_per_axis,
                thickness=feature_extraction_config["srp_thickness"]
            )
        if local_feature_extractor is not None:
            n_input_features = local_feature_extractor.n_output

        # 3. Create pairwise feature extractor
        self.use_pairwise_feature_extractor = pairwise_feature_extractor is not None
        if self.use_pairwise_feature_extractor:
            if pairwise_feature_extractor == "gcc_phat":
                pairwise_feature_extractor = GccPhat(self.sr, feature_extraction_config["n_dft"])
            elif pairwise_feature_extractor == "spatial_likelihood_grid":
                pairwise_feature_extractor = SpatialLikelihoodGrid(self.sr, self.n_likelihood_grid_points_per_axis)
            else:
                raise ValueError("pairwise_feature_extractor must be 'gcc_phat' or 'spatial_likelihood_grid'")

            n_input_features = pairwise_feature_extractor.n_output
    
        pairwise_feature_extractor = MetadataAwarePairwiseFeatureExtractor(
            n_input_features,
            pairwise_feature_extractor,
            is_metadata_aware,
            use_rt60_as_metadata
        )

        super().__init__(n_input_features, n_output_features, n_pairwise_features, local_feature_extractor,
                         pairwise_feature_extractor, pairwise_network_only,
                         activation, output_activation, init_layers, batch_norm,
                         dropout_rate, n_layers, SIGNAL_KEY)


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
