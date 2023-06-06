import hydra
import pytorch_lightning as pl
import torch.cuda as torch_gpu
# import torch.backends.mps as torch_mps

from hydra.utils import get_class
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    ModelCheckpoint, RichProgressBar, TQDMProgressBar,
    EarlyStopping
)

from gnn_ssl.models.base.utils import load_checkpoint

SAVE_DIR = "logs/"


class Trainer(pl.Trainer):
    def __init__(self, config):

        # 1. Load method (trainer type) specified in the config.yaml file
        method = get_class(config["class"])(config)

        accelerator = "cpu"
        if torch_gpu.is_available():
            accelerator = "cuda"
        # elif torch_mps.is_available():
        #     accelerator = "mps"
        # Currently disabled, as mps doesn't support fft
        
        training_config = config["training"]
        strategy = training_config["multi_gpu_strategy"]
        n_devices = torch_gpu.device_count()
        if (not training_config["multi_gpu"]) or (n_devices <= 1):
            # Strategy is only necessary when using multiple GPUs
            strategy = None
            n_devices = 1

        if training_config["progress_bar_type"] == "rich":
            progress_bar = RichProgressBar(
                refresh_rate=training_config["progress_bar_refresh_rate"])
        elif training_config["progress_bar_type"] == "tqdm":
            progress_bar = TQDMProgressBar(
                refresh_rate=training_config["progress_bar_refresh_rate"])

        # Create callbacks (Progress bar, early stopping and weight saving)
        callbacks=[
            progress_bar,
            ModelCheckpoint(monitor="validation_loss",
                            save_last=True,
                            filename='weights-{epoch:02d}-{validation_loss:.2f}'
            )
        ]
        early_stopping_config = training_config["early_stopping"]
        if early_stopping_config["enabled"]:
            callbacks.append(
                EarlyStopping(early_stopping_config["key_to_monitor"],
                              early_stopping_config["min_delta"],
                              early_stopping_config["patience_in_epochs"]
                )
            )

        super().__init__(
            max_epochs=training_config["n_epochs"],
            callbacks=callbacks,
            logger=[pl_loggers.TensorBoardLogger(save_dir=SAVE_DIR)],
            accelerator=accelerator,
            strategy=strategy,
            log_every_n_steps=25,
            check_val_every_n_epoch=training_config["check_val_every_n_epoch"],
            devices=n_devices,
        )

        ckpt_path = config["inputs_train"]["checkpoint_path"]
        if ckpt_path is not None:
            load_checkpoint(method.model, ckpt_path)
        
        self._method = method
        self.config = config

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._method, train_dataloaders,
                    val_dataloaders=val_dataloaders)

    def fit_multiple(self, train_dataloaders, val_dataloaders=None):
        """Fit the model sequentially for multiple datasets"""

        for i, train_dataloader in enumerate(train_dataloaders):
            val_dataloader = None
            if val_dataloaders is not None:
                val_dataloader = val_dataloaders[i]
            super().fit(self._method, train_dataloader,
                        val_dataloaders=val_dataloader)


    def test(self, test_dataloaders):
        super().test(self._method, test_dataloaders, ckpt_path="best")


def _create_torch_dataloaders(config):
    dataset_paths = config["inputs_train"]["dataset_paths"]
    dataloader = get_class(config["dataset"]["class"])

    batch_size = config["training"]["batch_size"]
    n_workers = config["training"]["n_workers"]
    training_dataloader = dataloader(config, dataset_paths["training"],
                                     batch_size=batch_size, n_workers=n_workers, shuffle=True)
    
    val_dataloader = None
    test_dataloader = None

    if dataset_paths["validation"]:
        val_dataloader = dataloader(config, dataset_paths["validation"],
                                    batch_size=batch_size, n_workers=n_workers)
    
    if dataset_paths["test"]:
        test_dataloader = dataloader(config, dataset_paths["test"],
                                     batch_size=batch_size, n_workers=n_workers)

    return (
        training_dataloader,
        val_dataloader,
        test_dataloader
    )


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 

    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    dataset_train, dataset_val, dataset_test = _create_torch_dataloaders(config)

    trainer = Trainer(config)

    if config["training"]["transfer_learning"]:
        trainer.fit_multiple(dataset_train, val_dataloaders=dataset_val)
    else:
        trainer.fit(dataset_train, val_dataloaders=dataset_val)

    trainer.test(dataset_test)


if __name__ == "__main__":
    main()
