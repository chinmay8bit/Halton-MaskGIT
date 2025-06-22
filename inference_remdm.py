import random
import torch

from pipeline_remdm import Pipeline, ReMDMScheduler
from Network.vq_model import VQ_models
from Network.transformer import Transformer

from plot_utils import show_images_grid

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

scheduler = ReMDMScheduler(
    schedule="cosine",
    remask_strategy="rescale",
    eta=0.05,
    temperature=0.8,
)

pipe = Pipeline(
    vqvae=vqvae,
    transformer=transformer,
    scheduler=scheduler,
    codeboook_size=codebook_size,
    mask_token_id=codebook_size,
    latent_height=input_size,
    latent_width=input_size,
    device=device,
    use_mixed_precision=True,
)

num_samples = 16
# goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
labels = [1, 7, 282, 604, 724, 179, 751, 404, 850] + [random.randint(0, 999) for _ in range(num_samples - 9)]

labels = [22] * 8

images = pipe(
    num_samples=len(labels),
    num_inference_steps=100,
    guidance_scale=2,
    labels=torch.tensor(labels).to(device),
    unconditional=False,
)

print(images.shape)

show_images_grid(images, save_file="output.png")