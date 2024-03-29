from omegaconf import OmegaConf
import torch

from torch.optim.lr_scheduler import MultiStepLR

from ..models.gnn_ssl import GnnSslNet

from ..metrics import Loss
from .base import BaseTrainer


class GnnSslNetTrainer(BaseTrainer):
    """This class abstracts the
       training/validation/testing procedures
       used for training a GnnSslNet
    """

    def __init__(self, config):
        super().__init__(config)

        self.rmse = Loss(config["targets"], mode="l2", dim=1)

    def _step(self, batch, batch_idx, log_model_output=False, log_labels=False):
        """This is the base step done on every training, validation and testing batch iteration.
        This had to be overriden since we are using multiple datasets
        """

        loss_vectors = []
        outputs = []
        targets = []
        n_mics = []
        
        # Loop for every microphone number
        for (x, y) in batch:
            n = x["signal"].shape[1]

            # 1. Compute model output and loss
            output = self.model(x)
            loss = self.loss(output, y, mean_reduce=False)

            outputs.append(output)
            targets.append(y)
            loss_vectors.append(loss)
            n_mics.append(n)

            # 2. RMSE
            rmse_error = self.rmse(output, y, mean_reduce=True)
            self.log(f"rmse_{n}_mics", rmse_error, on_step=True, prog_bar=False, on_epoch=False)

        output_dict = {}

        for i, n in enumerate(n_mics):
            loss_vector = loss_vectors[i]
            mean_loss = loss_vector.mean()
            output_dict[f"loss_vector_{n}"] = loss_vector.detach().cpu()
            output_dict[f"mean_loss_{n}"] = mean_loss.detach().cpu()

            # TODO: Add these to a callback
            # 2. Log model output
            if log_model_output:
                output_dict[f"model_outputs_{n}"] = outputs[i]
            

        output_dict["loss"] = torch.stack(loss_vectors).mean()

        return output_dict
