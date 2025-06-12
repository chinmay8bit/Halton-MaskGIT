import random
import torch

from pipeline import Pipeline, HaltonScheduler
from Network.vq_model import VQ_models
from Network.transformer import Transformer

device = torch.device('cuda')
codebook_size = 16384
img_size = 384
vae_scale_factor = 16
input_size = img_size // vae_scale_factor


# Initialize and load VQGAN model
vqvae = VQ_models['VQ-16'](codebook_size=codebook_size, codebook_embed_dim=8)
checkpoint = torch.load('./saved_networks/vq_ds16_c2i.pt', map_location="cpu", weights_only=False)
vqvae.load_state_dict(checkpoint["model"])
vqvae = vqvae.eval()

# Initialize and load MaskGIT transformer
hidden_dim, depth, heads = 1024, 24, 16
transformer = Transformer(
    input_size=input_size, nclass=1000, c=hidden_dim,
    hidden_dim=hidden_dim, codebook_size=codebook_size,
    depth=depth, heads=heads, mlp_dim=hidden_dim * 4,
    register=1, proj=1
)
checkpoint = torch.load("./saved_networks/ImageNet_384_large.pth", map_location='cpu', weights_only=False)
transformer.load_state_dict(checkpoint["model_state_dict"])
transformer = transformer.eval()

scheduler = HaltonScheduler(
    latent_height=input_size,
    latent_width=input_size,
    sm_temp_min=1,
    sm_temp_max=1.2,
    temp_pow=1,
    temp_warmup=0,
)

pipe = Pipeline(
    vqvae=vqvae,
    transformer=transformer,
    scheduler=scheduler,
    mask_token_id=codebook_size,
    latent_height=input_size,
    latent_width=input_size,
    device=device,
)

num_samples = 16
# goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
labels = [1, 7, 282, 604, 724, 179, 751, 404, 850] + [random.randint(0, 999) for _ in range(num_samples - 9)]

labels = [22] * 16

images = pipe(
    num_samples=len(labels),
    num_inference_steps=32,
    guidance_scale=2,
    labels=torch.tensor(labels).to(device),
    unconditional=False,
)

def show_images_grid(batch, nrow=4, padding=2):
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    # Create the grid
    grid = make_grid(batch, nrow=nrow, padding=padding)

    # Move the grid to CPU and convert to numpy
    grid = grid.permute(1, 2, 0).cpu().numpy()

    # Display the grid
    plt.figure(figsize=(nrow * 2, (len(batch) // nrow + 1) * 2))
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig("output_pipeline.png")
    plt.show()

show_images_grid(images)
