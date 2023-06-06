import matplotlib.pyplot as plt
import hydra
import pandas as pd
import seaborn as sns
import torch

from hydra.utils import get_class
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_ssl.models.base.utils import load_checkpoint

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _compute_batch_metrics(batch, batch_idx, models, metrics):
    x, y = batch[0]

    x = _dict_to_device(x, DEVICE)
    y = _dict_to_device(y, DEVICE)
    
    results = {}

    for model_name, model in models.items():
        y_hat = model(x)
        results[model_name] = {}
        for metric_name, metric in metrics.items():
            values = metric(y_hat, y)
            if isinstance(values, dict): # It computed multiple metrics
                results[model_name].update(values)
            else:
                results[model_name][metric_name] = values.detach().cpu()
    return results


def _evaluate_dataset(dataset_name: str, config: DictConfig, models: dict, metrics: dict, batch_size: int):
    print(f"Evaluating models on dataset '{dataset_name}'")
    # 1. Load evaluation dataset
    dataset_config = config["dataset"]
    inputs = config["inputs_eval"]

    dataloader = get_class(dataset_config["class"])
    dataset = dataloader(config, inputs["dataset_paths"][dataset_name], batch_size=batch_size)

    # Only evaluate on compatible models
    models = {
        model_name: model["model"] for model_name, model in models.items()
        if dataset_name in model["evaluate_on"]
    }

    metrics = _compute_metrics(dataset_name, dataset, models, metrics)

    return metrics


def load_models(config):
    evaluation_config = config["evaluation"]
    inputs = config["inputs_eval"]

    # 1. Load models
    models = {}

    for model_name, model_config in evaluation_config["models"].items():
        evaluate_on = model_config["evaluate_on"]
        model_config = model_config["model"]
        model = get_class(model_config["class"])(model_config)

        models[model_name] = {
            "model": model.eval(),
            "evaluate_on": evaluate_on
        }
    
    # 2. Load model checkpoints for trained methods (i.e. not classical signal processing ones)
    for model_name, checkpoint_path in inputs["checkpoint_paths"].items():
        if model_name in models:
            load_checkpoint(models[model_name]["model"], checkpoint_path)

    return models


def _load_metrics(config):
    metrics = {}
    for metric_name, metric_config in config["evaluation"]["metrics"].items():
        metric_class = metric_config["class"]
        metric_config = metric_config["config"] if "config" in metric_config else {}
        metric = get_class(metric_class)(**metric_config)
        metrics[metric_name] = metric

    return metrics


def _compute_metrics(dataset_name: str, dataloader: DataLoader, models: dict, metrics: dict):
    if not models:
        return pd.DataFrame()

    # 1. Compute metrics for all batches
    batch_metrics = []
    for i, batch in enumerate(tqdm(dataloader)):
        batch_metrics.append(
            _compute_batch_metrics(batch, i, models, metrics)
        )

    # 2. Group metrics for all batches
    metrics = _group_batch_metrics(batch_metrics, dataset_name)

    # 3. Compute aggregate metrics
    _plot_histograms(metrics, dataset_name)
    
    return metrics


def _group_batch_metrics(batch_metrics: dict, dataset_name: str):
    "Group batch metrics into a Pandas dataframe"
    metrics = {}
    df_metrics = []

    model_names = list(batch_metrics[0].keys())
    metric_names = list(batch_metrics[0][model_names[0]].keys())

    for model_name in batch_metrics[0].keys():
        metrics[model_name] = {}
        metric_names = batch_metrics[0][model_name].keys()
        for metric_name in metric_names:
            errors = torch.cat([result[model_name][metric_name] for result in batch_metrics])
            metrics[model_name][metric_name] = errors

            df = pd.DataFrame.from_dict({
                "value": errors.tolist(),
                "model": [model_name]*len(errors),
                "metric": [metric_name]*len(errors),
                "dataset": [dataset_name]*len(errors)} ,orient='index').transpose()
                
            df_metrics.append(df)

    return pd.concat(df_metrics)


def _plot_histograms(df: pd.DataFrame, dataset_name: str):
    """Plot one histogram per metric, comparing all models for a given dataset
    """

    # the first key in the dict is the model, the second is the metric
    
    metric_names = df["metric"].unique()

    # Create a plot for each metric
    for metric_name in metric_names:
        fig, ax = plt.subplots()
        ax.set_title(f"{metric_name}")
        df_metric = df[df["metric"] == metric_name]
        try:
            sns.histplot(
                df_metric, x="value",
                hue="model", multiple="layer",
                palette="Blues",
                ax=ax
            )
        except TypeError:
            continue

        ax.set_xlabel("Error (m)")
        ax.set_ylabel("Num. test cases")
        plt.savefig(f"outputs/{dataset_name}_{metric_name}.pdf")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):
    """Evaluate the model and produce histograms
    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    # 1. Load models and checkpoints
    models = load_models(config)

    # 2. Load metrics
    metrics = _load_metrics(config)

    # 3. Evaluate metrics for every model, for each dataset
    inputs = config["inputs_eval"]
    dataset_names = list(inputs["dataset_paths"].keys())
    batch_size = config["evaluation"]["batch_size"]

    results = []
    for dataset_name in dataset_names:
        dataset_results = _evaluate_dataset(dataset_name, config, models, metrics, batch_size)
        results.append(dataset_results)

    results = pd.concat(results, ignore_index=True)
    
    results.to_csv("results.csv")


def _dict_to_device(d, device):
    new_d = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            new_d[key] = value.to(device)
        elif isinstance(value, dict):
            new_d[key] = _dict_to_device(value, device)
    return d


if __name__ == "__main__":
    main()
