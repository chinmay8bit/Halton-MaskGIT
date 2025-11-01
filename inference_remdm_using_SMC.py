import random
import torch

# from pipeline_remdm_using_SMC import Pipeline, ReMDMScheduler
# from pipeline_remdm_using_SMC_hessian import Pipeline, ReMDMScheduler
from pipeline_remdm_using_SMC_reverse import Pipeline, ReMDMScheduler
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

def get_classfier_fn():
    from transformers import ViTFeatureExtractor, ViTForImageClassification
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
        return logits[torch.arange(len(labels)), labels].clamp_max(-0.0)
    return tmp_fn
classifier_fn = get_classfier_fn()


scheduler = ReMDMScheduler(
    schedule="cosine",
    remask_strategy="rescale",
    eta=0.2,
    temperature=0.9,
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

num_particles = 8
batch_p = 4
reward_fn = lambda images : classifier_fn(
    images, 
    torch.tensor(
        [1] * len(images)
    ).to(device)
)

images, _ = pipe(
    num_inference_steps=100,
    disable_progress_bar=True,
    # SMC paramters
    num_particles=num_particles,
    batch_p=batch_p,
    kl_coeff=0.5,
    tempering_gamma=0.05,
    reward_fn=reward_fn,
    verbose=True,
    # ess_threshold=0,
)

show_images_grid(images[:8], save_file="output_SMC.png")
