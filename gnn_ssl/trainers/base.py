import pytorch_lightning as pl
import torch

from hydra.utils import get_class
from torch.optim.lr_scheduler import MultiStepLR


class BaseTrainer(pl.LightningModule):
    """Class which abstracts interactions with Hydra
    and basic training/testing/validation conventions
    """

    def __init__(self, config):
        super().__init__()

        model, training = config["model"], config["training"]
        features, targets = config["features"], config["targets"]

        self.model = get_class(model["class"])(model)
        # self.feature_extractor = get_class(features["class"])(features)
        self.loss = get_class(targets["loss_class"])(targets)

        self.model_config =  model
        self.features_config = features
        self.targets_config = targets
        self.training_config = training

        self.log_step = self.training_config["log_step"]

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, epoch_type):

        x, y = batch[0] # 0 is to ignore the microphone array index
        
        # 1. Compute model output and loss
        x = self.forward(x)
        loss = self.loss(x, y, mean_reduce=True)

        self.log_dict({f"{epoch_type}_loss_step": loss})
        
        return {
            "loss": loss
        }

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")
  
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "validation")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
    
    def _epoch_end(self, outputs, epoch_type="train"):
        # 1. Compute epoch metrics
        outputs = _merge_list_of_dicts(outputs)
        tensorboard_logs = {
            f"{epoch_type}_loss": outputs["loss"].mean(),
            # f"{epoch_type}_std": outputs["loss"].std(),
            "step": self.current_epoch
        }

        print(tensorboard_logs)
        self.log_dict(tensorboard_logs)
    
    def training_epoch_end(self, outputs):
        self._epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, epoch_type="validation")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, epoch_type="test")

    def forward(self, x):
        return self.model(x)
        
    def fit(self, dataset_train, dataset_val):
        super().fit(self.model, dataset_train, val_dataloaders=dataset_val)

    def test(self, dataset_test, ckpt_path="best"):
        super().test(self.model, dataset_test, ckpt_path=ckpt_path)

    def configure_optimizers(self):
        lr = self.training_config["learning_rate"]
        decay_step = self.training_config["learning_rate_decay_steps"]
        decay_value = self.training_config["learning_rate_decay_values"]

        if self.training_config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif self.training_config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                            betas=self.training_config["betas"])
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]


def _merge_list_of_dicts(list_of_dicts):
    """Function used at the end of an epoch.
    It is used to merge together the many vectors generated at each step
    """

    result = {}

    def _add_to_dict(key, value):
        if len(value.shape) == 0: # 0-dimensional tensor
            value = value.unsqueeze(0)

        if key not in result:
            result[key] = value
        else:
            result[key] = torch.cat([
                result[key], value
            ])
    
    for d in list_of_dicts:
        for key, value in d.items():
            _add_to_dict(key, value)

    return result