# %%
import torch
from Utils.utils import load_args_from_file
from Utils.viz import show_images_grid
from huggingface_hub import hf_hub_download

from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

# %%
config_path = "Config/base_cls2img.yaml"        # Path to your config file
args = load_args_from_file(config_path)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download the VQGAN from LlamaGen 
hf_hub_download(repo_id="FoundationVision/LlamaGen", 
                filename="vq_ds16_c2i.pt", 
                local_dir="./saved_networks/")

# Download the MaskGIT
# hf_hub_download(repo_id="llvictorll/Halton-Maskgit", 
#                 filename="ImageNet_384_large.pth", 
#                 local_dir="./saved_networks/")
# hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", 
#                 filename="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth", 
#                 local_dir="./saved_networks/")

# Initialisation of the model
model = MaskGIT(args)

# select your scheduler
sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=2,
                        sched_pow=2, step=32, randomize=True, top_k=-1)

# [goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner]
# labels = [1, 7, 282, 604, 724, 179, 751, 404] 
labels = None

gen_images = sampler(trainer=model, nb_sample=16, labels=labels, verbose=True)[0]
# %%
show_images_grid(gen_images)

print(gen_images.shape)

def classify(image):
    from transformers import ViTFeatureExtractor, ViTForImageClassification
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-384')
    if feature_extractor.do_normalize:
        mean = torch.tensor(feature_extractor.image_mean, device=image.device).view(1, 3, 1, 1)
        std  = torch.tensor(feature_extractor.image_std, device=image.device).view(1, 3, 1, 1)
        image = (image - mean) / std
    
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-384').to(image.device)
    outputs = model(image)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_indices = logits.argmax(-1)
    for predicted_class_idx in predicted_class_indices:
        print("Predicted class:", predicted_class_idx.item(), model.config.id2label[predicted_class_idx.item()])
    return logits.max(dim=-1)[0]


with torch.enable_grad():
    images = gen_images.clone()
    images = (images + 1) / 2.0 
    
    images = images.requires_grad_(True)
    
    logits = classify(images)
    print(logits.shape)
    
    tmp_approx_guidance = torch.autograd.grad(
                            outputs=logits, 
                            inputs=images,
                            grad_outputs=torch.ones_like(logits)
                        )[0].detach()
    print(tmp_approx_guidance.shape)
    print(tmp_approx_guidance.sum())


def show_images_grid_2(batch, nrow=4, padding=2):
    """
    Displays a batch of images concatenated into a single grid using PyTorch's make_grid.

    Args:
        batch (torch.Tensor): Batch of images, shape (B, C, H, W), with values in range [-1, 1].
        nrow (int): Number of images in each row of the grid.
        padding (int): Padding between images in the grid.
    """
    # # Unnormalize the tensor from [-1, 1] to [0, 1]
    # batch = (batch + 1) / 2.0

    # Clamp to ensure all values are in [0, 1]
    # batch = batch.clamp(0, 1)
    
    batch = batch.mean(dim=1, keepdim=True)
    
    batch = batch - batch.flatten(start_dim=1).min(dim=-1)[0][:, None, None, None]
    batch = batch / batch.flatten(start_dim=1).max(dim=-1)[0][:, None, None, None]

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
    plt.savefig("grad.png")
    plt.show()

show_images_grid_2(tmp_approx_guidance.detach().cpu())