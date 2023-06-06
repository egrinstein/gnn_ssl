# Plots for the interspeech2023 submission

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_plots(results_csv_path):
    results = pd.read_csv(results_csv_path)

    # TODO: Make lots of "group by" instead enumerating specifics.
    # Create a "dataset_type" parameter and a "model_type" parameter
    # Which would allow grouping of datasets of the same class (4 mics, for example)
    # and methods of the same class (CRNN for example).

    dataset_names = list(results["dataset"].unique())
    model_names = list(results["model"].unique())
    
    # 1. Results for individual datasets
    for dataset_name in dataset_names:
        df = results[results["dataset"] == dataset_name]
        _plot_histogram(df, dataset_name, "model")

    # 2. Results for number of mics
    for dataset_group in ["4 mics", "6 mics"]:
        df = results[results["dataset"].str.contains(dataset_group)]
        _plot_histogram(df, dataset_group, "model")

    # 3. Results for reverb/recorded
    for dataset_group in ["reverb", "recorded"]:
        df = results[results["dataset"].str.contains(dataset_group)]
        _plot_histogram(df, dataset_group, "model")
    
    # 4. Results for individual methods
    for model_name in model_names:
        df = results[results["model"] == model_name]
        _plot_histogram(df, model_name, "dataset")


# def _plot_histogram(df, dataset_name, hue, ax=None):
#     if not ax:
#         fig, ax = plt.subplots()
#     sns.histplot(
#         df, x="value",
#         hue=hue, multiple="dodge",
#         palette="Blues",
#         ax=ax,
#         bins=15
#     )

#     ax.set_title(dataset_name)
#     ax.set_xlabel("Error (m)")
#     ax.set_ylabel("Num. test cases")
#     plt.savefig(f"outputs/{dataset_name}.pdf")


def _plot_histogram(df, dataset_name, group_by, ax=None):
    fig, ax = plt.subplots()
    cmap = mpl.cm.get_cmap('Set2')
    ax.set_title(f"{group_by}")

    groups = df[group_by].unique()
    n_groups = len(groups)

    for i, key in enumerate(groups):
        color = cmap((i+1)/n_groups)
        values = df[df[group_by] == key]["value"]
        counts, bins = np.histogram(values, bins=30)
        ax.stairs(counts, bins, label=key, color=color)
        try:
            mean = values.mean()
            ax.axvline(x=mean, color=color, alpha=0.2, linestyle="--",
                        label="mean={:.2f} m".format(mean))
        except RuntimeError:
            pass
    ax.legend()
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Num. test cases")
    plt.savefig(f"outputs/{dataset_name}.pdf")

def make_tables(results_csv_path):
    results = pd.read_csv(results_csv_path)
    df_mean = []

    dataset_names = results["dataset"].unique()
    model_names = results["model"].unique()

    for dataset_name in dataset_names:
        df_dataset = results[results["dataset"] == dataset_name]
        for model_name in model_names:
            df_model = df_dataset[df_dataset["model"] == model_name]
            df_mean.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Value": df_model["value"].mean(),
                "Std.": df_model["value"].std()
            })

    df_mean = pd.DataFrame(df_mean)
    df_mean.to_csv("means.csv")


if __name__ == "__main__":
    make_plots("results.csv")
    make_tables("results.csv")
