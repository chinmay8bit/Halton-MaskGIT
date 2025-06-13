from typing import Optional

import math
import torch
from tqdm import tqdm


from Sampler.halton_sampler import HaltonSampler



class HaltonScheduler:
    def __init__(
        self,
        latent_height: int,
        latent_width: int,
        sm_temp_min: float = 1,
        sm_temp_max: float = 1,
        temp_pow=1,
        top_k: int = -1,
        temp_warmup: int = 0,
    ):
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.sm_temp_min = sm_temp_min
        self.sm_temp_max = sm_temp_max
        self.temp_pow = temp_pow
        self.top_k = top_k
        self.temp_warmup_till = temp_warmup
        
        assert self.latent_height == self.latent_width
        self.basic_halton_mask = HaltonSampler.build_halton_mask(self.latent_height)
        self.latent_size = self.latent_height * self.latent_width
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        # Linearly interpolate the temperature over the sampling steps
        self.temperature = torch.linspace(self.sm_temp_min, self.sm_temp_max, self.num_inference_steps)
    
    def initialize_halton_masks(self, nb_sample, randomize=True):
        # Randomizing the mask sequence if enabled
        if randomize:
            randomize_mask = torch.randint(0, self.latent_size, (nb_sample,))
            self.halton_mask = torch.zeros(nb_sample, self.latent_size, 2, dtype=torch.long)
            for i_h in range(nb_sample):
                rand_halton = torch.roll(self.basic_halton_mask.clone(), randomize_mask[i_h].item(), 0) # type: ignore
                self.halton_mask[i_h] = rand_halton
        else:
            self.halton_mask = self.basic_halton_mask.clone().unsqueeze(0).expand(nb_sample, self.latent_size, 2)
        self.prev_r = 0
    
    def get_temperature(self, step):
        temp = self.temperature[step] ** self.temp_pow
        if step < self.temp_warmup_till:
            temp *= 0.5  # Reduce temperature during warmup
        return temp
    
    def sample(
        self,
        logits: torch.Tensor,
        latents: torch.Tensor,
        step: int,
    ):
        nb_sample = len(latents)
        
        # Compute the number of tokens to predict
        ratio = ((step + 1) / self.num_inference_steps)
        r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
        r = int(r * (self.latent_size))
        r = max(step + 1, r)
        
        # Construct the mask for the current step
        _mask = self.halton_mask.clone()[:, self.prev_r:r]
        mask = torch.zeros(nb_sample, self.latent_height, self.latent_width, dtype=torch.bool, device=logits.device)
        for i_mask in range(nb_sample):
            mask[i_mask, _mask[i_mask, :, 0], _mask[i_mask, :, 1]] = 1
        self.prev_r = r
        
        # Choose softmax temperature
        temp = self.get_temperature(step)
        
        # Compute probabilities using softmax
        prob = torch.softmax(logits * temp, -1)
        
        if self.top_k > 0:# Apply top-k filtering
            top_k_probs, top_k_indices = torch.topk(prob, self.top_k, dim=-1)              # (B,H,W,k)
            top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)                           # normalize
            # B, H, W, k = top_k_probs.shape
            next_token_index = torch.multinomial(
                top_k_probs.view(-1, self.top_k), num_samples=1
            ).view(nb_sample, self.latent_height, self.latent_width, 1)                    # (B,H,W,1)
            pred_code = top_k_indices.gather(dim=-1, index=next_token_index).squeeze(-1)  
        else:
            # Sample from the categorical distribution
            pred_code = torch.distributions.Categorical(probs=prob).sample()
        
        # Update code with new predictions
        new_latents = torch.where(
            mask,
            pred_code,
            latents,
        )
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
        self.scheduler.initialize_halton_masks(num_samples, randomize=True)
        
        logits = torch.zeros((*latents.shape, self.codebook_size), device=self._execution_device)
        
        #5. Inference steps
        bar = range(num_inference_steps) if disable_progress_bar else tqdm(range(num_inference_steps), leave=False)
        for i in bar:
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
        return logits[..., :self.codebook_size]
    
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
        return logits[..., :self.codebook_size]
    
    def decode_latents(self, latents):
        images = self.vqvae.decode_code(latents)
        images = torch.clamp(images, -1, 1)
        images = (images + 1.0) / 2.0
        return images
    