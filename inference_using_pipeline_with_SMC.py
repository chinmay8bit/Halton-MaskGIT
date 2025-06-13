import random
import torch

from transformers import ViTFeatureExtractor, ViTForImageClassification

from pipeline_using_SMC import Pipeline, HaltonScheduler
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

def get_classfier_fn():
    # Intialize reward models
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-384')
    classifier =  ViTForImageClassification.from_pretrained('google/vit-base-patch32-384').to(device) # type: ignore
    def tmp_fn(images, labels):
        if feature_extractor.do_normalize:
            mean = torch.tensor(feature_extractor.image_mean, device=images.device).view(1, 3, 1, 1)
            std  = torch.tensor(feature_extractor.image_std, device=images.device).view(1, 3, 1, 1)
            images = (images - mean) / std
        logits = classifier(images).logits
        logits = torch.log_softmax(logits, dim=-1)
        return logits[torch.arange(len(labels)), labels]
    return tmp_fn
classifier_fn = get_classfier_fn()


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
    codeboook_size=codebook_size,
    mask_token_id=codebook_size,
    latent_height=input_size,
    latent_width=input_size,
    device=device,
    # use_mixed_precision=True,
)

num_samples = 16
# goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
labels = [1, 7, 282, 604, 724, 179, 751, 404, 850] + [random.randint(0, 999) for _ in range(num_samples - 9)]

num_particles = 15
batch_p = 3
labels = [1] * batch_p
reward_fn = lambda images : classifier_fn(images, torch.tensor(labels).to(device))

images = pipe(
    num_inference_steps=100,
    disable_progress_bar=True,
    # SMC paramters
    num_particles=num_particles,
    batch_p=batch_p,
    kl_coeff=0.1,
    tempering_gamma=0.03,
    reward_fn=reward_fn,
    verbose=True,
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
    plt.savefig("output_pipeline_SMC.png")
    plt.show()

show_images_grid(images)
