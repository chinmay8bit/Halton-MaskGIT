from torchvision.utils import make_grid
import matplotlib.pyplot as plt


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
