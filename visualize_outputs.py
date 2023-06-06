import hydra

from hydra.utils import get_class
from omegaconf import DictConfig

from gnn_ssl.visualization import create_visualizations
from evaluate import load_models


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):
    """Produce visualization of the model output
    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    # 1. Load evaluation dataset
    dataloader = get_class(config["dataset"]["class"])
    inputs = config["inputs_eval"]

    dataset_names = list(inputs["dataset_paths"].keys()) 
    dataset_path = inputs["dataset_paths"][dataset_names[0]] # Only select first dataset right now, change this in future
    dataset_test = dataloader(config, dataset_path, shuffle=False)

    # 2. Load models
    models = load_models(config)

    # 3. Load loss function
    loss = get_class(config["targets"]["loss_class"])(config["targets"])
    
    # Compute outputs for a single batch
    x, y = batch = next(iter(dataset_test))[0]

    model_outputs = {
        model_name: model["model"](x)
        for model_name, model in models.items()
    }

    create_visualizations(model_outputs, y, loss, plot_target=False)
    

if __name__ == "__main__":
    main()
