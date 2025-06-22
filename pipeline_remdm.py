from typing import Optional

import math
import torch
from tqdm import tqdm

import torch.nn.functional as F

from plot_utils import show_images_grid


class ReMDMScheduler:
    def __init__(
        self,
        schedule,
        remask_strategy,
        eta,
        temperature=1.0,
    ):
        self.schedule = schedule
        self.remask_strategy = remask_strategy
        self.eta = eta 
        self.temperature = temperature

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        if self.schedule == "linear":
            self.alphas = 1 - torch.linspace(0, 1, num_inference_steps + 1)
        elif self.schedule == "cosine":
            self.alphas = 1 - torch.cos((torch.pi/2) * (1 - torch.linspace(0, 1, num_inference_steps + 1)))
        else:
            raise ValueError(f"unknown masking schedule {self.schedule}")

    def sample(
        self,
        logits: torch.Tensor,
        latents: torch.Tensor,
        step: int,
    ):
        B, H, W, C = logits.shape
        assert latents.shape == (B, H, W)
        
        logits = logits.reshape(B, H*W, C)
        latents = latents.reshape(B, H*W)
        
        t = self.num_inference_steps - step
        s = t - 1
        
        alpha_t = self.alphas[t]
        alpha_s = self.alphas[s]
        sigma_t_max = torch.clamp_max((1 - alpha_s) / alpha_t, 1.0)
        if self.remask_strategy == "max_cap":
            sigma_t = torch.clamp_max(sigma_t_max, self.eta)
        elif self.remask_strategy == "rescale":
            sigma_t = sigma_t_max * self.eta
        else:
            raise ValueError(f"unknown masking schedule {self.remask_strategy}")
        
        # z_t != m
        x_theta = F.one_hot(latents, num_classes=C).float()
        logits_z_t_neq_m = (
            torch.log(x_theta) +
            torch.log(1 - sigma_t)
        )
        logits_z_t_neq_m[..., C-1] = (
            torch.log(sigma_t)
        )
        
        # z_t = m
        log_x_theta = (logits / self.temperature).log_softmax(dim=-1)
        logits_z_t_eq_m = (
            log_x_theta + 
            torch.log((alpha_s - (1 - sigma_t) * alpha_t) / (1 - alpha_t))
        )
        logits_z_t_eq_m[..., C-1] = (
            torch.log((1 - alpha_s - sigma_t * alpha_t) / (1 - alpha_t))
        )
        
        z_t_neq_m = (latents != C-1)
        p_theta_logits = torch.where(
            z_t_neq_m.unsqueeze(-1).expand(-1, -1, C),
            logits_z_t_neq_m,
            logits_z_t_eq_m,
        )
        assert torch.allclose(torch.exp(p_theta_logits).sum(dim=-1), torch.ones(B, H*W, device=logits.device))
        
        new_latents = torch.distributions.Categorical(logits=p_theta_logits).sample()
        new_latents = new_latents.reshape(B, H, W)
        
        print("Masked:", (new_latents == C-1).sum(dim=(1, 2)))
        return new_latents


class Pipeline:
    def __init__(
        self, 
        vqvae, 
        transformer, 
        scheduler, 
        codeboook_size: int,
        mask_token_id: int, 
        latent_height: int, 
        latent_width: int, 
        device: torch.device = torch.device('cuda'),
        use_mixed_precision: bool = False,
    ):
        self.vqvae = vqvae.to(device)
        self.transformer = transformer.to(device)
        self.scheduler = scheduler
        self.codebook_size = codeboook_size
        self.mask_token_id = mask_token_id
        self.latent_height = latent_height
        self.latent_width = latent_width
        self._execution_device = device
        self.use_mixed_precision = use_mixed_precision
    
    @torch.no_grad()
    def __call__(
        self,
        num_samples: int = 1,
        batch_size: Optional[int] = None,
        num_inference_steps: int = 48,
        guidance_scale: Optional[float] = None,
        labels: Optional[torch.Tensor] = None,
        unconditional: bool = True,
        disable_progress_bar = False,
    ):
        #1. Check input
        if unconditional == False:
            assert guidance_scale is not None
            assert labels is not None and len(labels) == num_samples
                        
        #2. Set batch size
        if batch_size is None or batch_size > num_samples:
            batch_size = num_samples
        
        #3. Set up intial latents
        latents_shape = (num_samples, self.latent_height, self.latent_width)
        latents = torch.full(
            latents_shape, self.mask_token_id, dtype=torch.long, device=self._execution_device # type: ignore
        )
        
        #4. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        logits = torch.zeros((*latents.shape, self.codebook_size + 1), device=self._execution_device)
        
        #5. Inference steps
        bar = range(num_inference_steps) if disable_progress_bar else tqdm(range(num_inference_steps), leave=False)
        for i in bar:
            imgs = []
            for idx in range(math.ceil(num_samples / batch_size)):
                #5.1 Get logits
                batch_latents = latents[batch_size*idx : batch_size*(idx+1)].clone()
                if unconditional:
                    tmp_logits = self.get_unconditional_logits(batch_latents)
                else:
                    assert labels is not None
                    batch_labels = labels[batch_size*idx : batch_size*(idx+1)]
                    tmp_logits = self.get_logits(batch_latents, batch_labels, guidance_scale)
                logits[batch_size*idx : batch_size*(idx+1)] = tmp_logits
                imgs.append(
                    self.decode_latents(
                        self.get_pred_original_sample(tmp_logits, batch_latents)
                    )
                )
            show_images_grid(torch.cat(imgs, dim=0), save_file="output.png")
                
            #5.2 Update latents
            latents = self.scheduler.sample(
                logits,
                latents,
                i,
            )
        
        #6. Decode latents to get images
        images = []
        for idx in range(math.ceil(num_samples / batch_size)):
            batch_latents = latents[batch_size*idx : batch_size*(idx+1)]
            images.append(
                self.decode_latents(batch_latents)
            )
        images = torch.cat(images, dim=0)
        return images
                    
    def get_unconditional_logits(self, latents):
        nb_sample = len(latents)
        drop = torch.ones(nb_sample, dtype=torch.bool).to(self._execution_device)
        dummy_labels = torch.zeros(nb_sample, dtype=torch.long, device=self._execution_device)
        with torch.autocast(device_type="cuda", enabled=self.use_mixed_precision):
            logits = self.transformer(latents, dummy_labels, drop)
        logits = logits.float()
        logits = logits.view(nb_sample, self.latent_height, self.latent_width, -1)
        logits[..., self.codebook_size] = -torch.inf
        return logits
    
    def get_logits(self, latents, labels, guidance_scale):
        nb_sample = len(latents)
        drop = torch.ones(nb_sample, dtype=torch.bool).to(self._execution_device)
        with torch.autocast(device_type="cuda", enabled=self.use_mixed_precision):
            logits = self.transformer(
                torch.cat([latents, latents], dim=0),
                torch.cat([labels, labels], dim=0),
                torch.cat([~drop, drop], dim=0)
            )
        logits = logits.float()
        logits_c, logits_u = torch.chunk(logits, 2, dim=0)
        logits = (1 + guidance_scale) * logits_c - guidance_scale * logits_u
        logits = logits.view(nb_sample, self.latent_height, self.latent_width, -1)
        logits[..., self.codebook_size] = -torch.inf
        return logits
    
    def decode_latents(self, latents):
        images = self.vqvae.decode_code(latents)
        images = torch.clamp(images, -1, 1)
        images = (images + 1.0) / 2.0
        return images
    
    def get_pred_original_sample(
        self,
        logits: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:

        vocab_size = logits.shape[-1]
        codebook_size, height, width = self.codebook_size, self.latent_height, self.latent_width
        batch_size = len(sample)
        
        sample = sample.reshape(batch_size, height * width)
        logits = logits.reshape(batch_size, height * width, vocab_size)

        pred_original_sample = torch.distributions.Categorical(logits=logits).sample()
        
        pred_original_sample = torch.where(
            sample == vocab_size - 1,
            pred_original_sample,
            sample
        )
        assert (pred_original_sample != vocab_size - 1).all()
        
        pred_original_sample = pred_original_sample.reshape(batch_size, height, width)
        return pred_original_sample