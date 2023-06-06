from gnn_ssl.metrics import NormLoss
from .base import BaseTrainer


class PairwiseNeuralSrpTrainer(BaseTrainer):
    def _step(self, batch, batch_idx, epoch_type):
        x, y = batch[0] # 0 is because the current dataloader is multi-microphone
        
        return super()._step((x, y), batch_idx, epoch_type)


class NeuralSrpTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.norm = NormLoss("l2")

    def _step(self, batch, batch_idx, epoch_type):
        x, y = batch[0] # 0 is because the current dataloader is multi-microphone
        if self.targets_config["type"] == "source_coordinates":
            n_output_coords = self.targets_config["n_output_coordinates"]
            y["source_coordinates"] = y["source_coordinates"][:, 0, :n_output_coords] # Only select first source

        return super()._step((x, y), batch_idx, epoch_type)
