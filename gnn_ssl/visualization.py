import matplotlib.pyplot as plt

from pysoundloc.pysoundloc.visualization import plot_grid


def create_visualizations(model_outputs, y, loss, plot_target=False):
    # This function is specific for the NeuralSrp method

    model_names = list(model_outputs.keys())
    n_models = len(model_names)
    batch_size = model_outputs[model_names[0]]["grid"].shape[0]

    mic_coords = y["mic_coordinates"][:, :, :2]
    room_dims = y["room_dims"][..., :2]
    source_coords = y["source_coordinates"][..., :2]
    
    target_grids = loss.grid_generator(
            room_dims,
            source_coords,
            mic_coords,
    )

    n_plots = n_models
    if plot_target:
        n_plots +=1

    for i in range(batch_size):
        fig, axs = plt.subplots(nrows=n_plots, figsize=(5, 5))

        # Plot model outputs
        for j, model_name in enumerate(model_names):
            plot_grid(model_outputs[model_name]["grid"][i].detach(),
                      room_dims[i], source_coords=source_coords[i],
                      microphone_coords=mic_coords[i], log=False, ax=axs[j])
            axs[j].set_title(model_name)
            if j < n_plots - 1:
                axs[j].get_xaxis().set_visible(False)

        # Plot target
        if plot_target:
            plot_grid(target_grids[i], room_dims[i], source_coords=source_coords[i],
                    microphone_coords=mic_coords[i], log=False, ax=axs[n_models])
            axs[n_models].set_title("Target grid")

        plt.tight_layout()
        plt.savefig(f"{i}.pdf")
