import torch
from copy import deepcopy


def format_metadata(metadata, use_rt60_as_metadata=False):
    mic_coords = metadata["mic_coordinates"]

    if len(mic_coords.shape) == 3: # Multi device
        mic_coords = mic_coords[0]

    room_dims = metadata["room_dims"]

    m = {
        "local": {
            "mic_coordinates": mic_coords
        },
        "global": {
            "room_dims": room_dims,
        }
    }
    if use_rt60_as_metadata:
        m["global"]["rt60"] = torch.Tensor([metadata["rt60"]])

    return m


def flatten_metadata(metadata, metadata_dim=3):
    """Transform the metadata dictionary where the values are batches of metadata
        into a batch of 1D torch vectors.
    Args:
        metadata (dict): Dictionary containing the keys ["local"]["mic_coords"]
                         ["global"]["room_dims"] and ["global"]["rt60"] (optional)
        metadata_dim (int): Whether to give 3D or 2D mic coordinates and room dims 

    Returns:
        torch.Tensor: a matrix with flattened metadata for each batch
    """

    mic_coords = metadata["local"]["mic_coordinates"][:, :, :metadata_dim]
    room_dims = metadata["global"]["room_dims"][:, :metadata_dim]


    flattened_metadata = torch.cat([
        mic_coords.flatten(start_dim=1),
        room_dims
    ], dim=1)
    if "rt60" in metadata["global"].keys():
        flattened_metadata = torch.cat([
            flattened_metadata,
            metadata["global"]["rt60"]], dim=1
        )
    
    return flattened_metadata


def filter_local_metadata(metadata, idxs):
    result = deepcopy(metadata)
    result["local"]["mic_coordinates"] = result["local"]["mic_coordinates"][:, idxs] 

    return result
