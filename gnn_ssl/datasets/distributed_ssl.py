import torch

from torch.utils.data import DataLoader

from sydra.sydra.dataset import SydraDataset
from ..feature_extractors.metadata import format_metadata


class DistributedSslDataset(SydraDataset):
    def __init__(self, dataset_dir, config,
                 trim_signals_mode="random", flatten_metadata=True):
        super().__init__(dataset_dir, config["sr"],
                         config["n_input_seconds"], trim_signals_mode)

        self.use_rt60_as_metadata = config["use_rt60_as_metadata"]
        self.flatten_metadata = flatten_metadata

    def __getitem__(self, index):

        (x, y) = super().__getitem__(index)
        # 1. Add metadata to input

        # Collapse channels and arrays
        x = x.flatten(start_dim=0, end_dim=1)
        x = {
            "signal": x,
            "metadata": format_metadata(y, self.use_rt60_as_metadata)
        }

        if len(y["mic_coordinates"].shape) == 3:
            # Array
            y["mic_coordinates"] = y["mic_coordinates"][0]
        
        if len(y["source_coordinates"].shape) == 1:
            # Expand to multi-source
            y["source_coordinates"] = y["source_coordinates"].unsqueeze(0)

        y = {
            "source_coordinates": y["source_coordinates"],
            "mic_coordinates": y["mic_coordinates"],
            "room_dims": y["room_dims"],
        }

        return (x, y)


class DistributedSslDataLoader(DataLoader):
    def __init__(self, config, dataset_path, shuffle=False,
                 batch_size=16, n_workers=1, **kwargs):

        self.config = config
        self.dataset_path = dataset_path
        if type(dataset_path) == str:
            dataset_path = [dataset_path] # Convert to a 1-element list

        dataset_config = config["dataset"]
        datasets = [
            DistributedSslDataset(d, dataset_config)
            for d in dataset_path
        ]

        dataset = ConcatDataset(datasets)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
            num_workers=n_workers,
        )


# Source: https://forums.pytorchlightning.ai/t/how-to-use-multiple-train-dataloaders-with-different-lengths/214/2
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
