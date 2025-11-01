from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images_grid(batch, nrow=4, padding=2, save_file=None):
    # Create the grid
    grid = make_grid(batch, nrow=nrow, padding=padding)

    # Move the grid to CPU and convert to numpy
    grid = grid.permute(1, 2, 0).cpu().numpy()

    # Display the grid
    plt.figure(figsize=(nrow * 2, (len(batch) // nrow + 1) * 2))
    plt.imshow(grid)
    plt.axis("off")
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_histogram(x, save_file=None, bins=30,
                   title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plots a histogram of the values in an array or tensor.

    Parameters
    ----------
    x : array-like
        The data to histogram (e.g. a 1D NumPy array, or torch.Tensor.flatten()).
    save_file : str, optional
        Path where to save the figure. If None, the plot is shown interactively.
    bins : int, optional
        Number of bins in the histogram (default: 30).
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x‐axis.
    ylabel : str, optional
        Label for the y‐axis.
    """
    # Convert to NumPy array and flatten
    data = np.array(x).flatten()
    
    # Create figure and histogram
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        

def plot_errors(approximation_errors, rewards_trace):
    # approximation_errors = torch.cat(approximation_errors)
    # rewards_trace = torch.cat(rewards_trace)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(approximation_errors, 'r-', label='error')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    # ax1.set_yscale('log')
    ax1.set_ylabel('Error (log)', color='r')
    ax2.plot(rewards_trace, 'b-', label='rewards', linewidth=0.5)
    ax2.set_ylabel('Rewards', color='b')
    fig.tight_layout()
    plt.savefig("approx_error.png")
    plt.close()
