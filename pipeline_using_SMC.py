from typing import Optional, Union, Callable

import math
import torch
from tqdm import tqdm
import torch.nn.functional as F


from Sampler.halton_sampler import HaltonSampler
from smc_utils import compute_ess_from_log_w, resampling_function, adaptive_tempering


def sum_masked_logits(
    logits: torch.Tensor,
    preds: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Sum the logits corresponding to the predicted classes at positions where mask is True.

    Args:
        logits (Tensor): shape (B, H, W, C), model output logits.
        preds (LongTensor): shape (B, H, W), predicted class indices in [0, C-1].
        mask (BoolTensor): shape (B, H, W), mask indicating which positions to include.

    Returns:
        Tensor: a 1D tensor of shape (B,) containing the sum of masked logits per sample.
    """
    # 1) Gather the logit at each predicted class
    pred_logits = logits.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
    # 2) Zero out unmasked positions and sum over spatial dims
    masked_logits = pred_logits * mask.to(pred_logits.dtype)
    return masked_logits.sum(dim=(1, 2))


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
    
    def sample_with_approx_guidance(
        self,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
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
        
        logits_with_approx_guidance = logits + approx_guidance
        
        # Choose softmax temperature
        temp = self.get_temperature(step)
        
        # Compute probabilities using softmax
        prob = torch.softmax(logits_with_approx_guidance * temp, -1)
        
        # Sample from the categorical distribution
        preds = torch.distributions.Categorical(probs=prob).sample()
        
        # Update code with new predictions
        new_latents = torch.where(
            mask,
            preds,
            latents,
        )
        
        # Calculate prob proposal and prob diffusion
        logits = logits.log_softmax(dim=-1)
        logits_with_approx_guidance = logits_with_approx_guidance.log_softmax(dim=-1)
        
        log_prob_proposal = sum_masked_logits(logits_with_approx_guidance, preds, mask)
        log_prob_diffusion = sum_masked_logits(logits, preds, mask)
        
        return new_latents, log_prob_proposal, log_prob_diffusion


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
        num_inference_steps: int = 48,
        disable_progress_bar = False,
        # SMC parameters
        num_particles: int = 4,
        batch_p: int = 1, # number of particles to run parallely
        resample_strategy: str = "ssp",
        ess_threshold: float = 0.5,
        tempering: str = "schedule",
        tempering_schedule: Union[float, int, str] = "exp",
        tempering_gamma: float = 1.,
        tempering_start: float = 0.,
        reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, # Ex) lambda images: _fn(images, prompts.repeat_interleave(batch_p, dim=0), metadata.repeat_interleave(batch_p, dim=0))
        kl_coeff: float = 1.,
        verbose: bool = False # True for debugging SMC procedure
    ):
        #1. Check input
        ...
                        
        #2. Set batch size
        batch_size = min(batch_p, num_particles)
        
        #3. Set up intial latents
        latents_shape = (num_particles, self.latent_height, self.latent_width)
        latents = torch.full(
            latents_shape, self.mask_token_id, dtype=torch.long, device=self._execution_device # type: ignore
        )
        
        #4. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.initialize_halton_masks(num_particles, randomize=True)
        
        # Intialize variables for SMC sampler
        logits = torch.zeros((*latents.shape, self.codebook_size), device=self._execution_device) # type: ignore
        approx_guidance = torch.zeros((*latents.shape, self.codebook_size), device=self._execution_device) # type: ignore 
        log_w = torch.zeros(latents.shape[0], device=self._execution_device)
        log_prob_diffusion = torch.zeros(latents.shape[0], device=self._execution_device)
        log_prob_proposal = torch.zeros(latents.shape[0], device=self._execution_device)
        log_twist_func = torch.zeros(latents.shape[0], device=self._execution_device)
        log_twist_func_prev = torch.zeros(latents.shape[0], device=self._execution_device)
        rewards = torch.zeros(latents.shape[0], device=self._execution_device)
        resample_fn = resampling_function(resample_strategy=resample_strategy, ess_threshold=ess_threshold)
        scale_factor = 0.
        min_scale_next = 0.
        
        kl_coeff = torch.tensor(kl_coeff, device=self._execution_device).to(torch.float32) # type: ignore
        lookforward_fn = lambda r: r / kl_coeff
        
        start = int(num_inference_steps * tempering_start)
        
        def _calc_guidance():
            assert latents is not None
            if (i >= start):
                for idx in range(math.ceil(num_particles / batch_size)):
                    with torch.enable_grad():
                        latents_one_hot = F.one_hot(
                            latents[batch_size*idx : batch_size*(idx+1)], 
                            num_classes=self.codebook_size + 1 # type: ignore
                        ).float().requires_grad_(True)

                        tmp_logits = self.get_unconditional_logits(latents_one_hot)
                
                        tmp_pred_original_sample: torch.Tensor = self.get_pred_original_sample(
                            logits=tmp_logits,
                            sample_one_hot=latents_one_hot,
                            use_continuous_formualtion=True,
                        )
                        
                        tmp_pred_original_sample_decoded = self.decode_one_hot_latents(tmp_pred_original_sample)
                        
                        # Calculate rewards
                        tmp_rewards = reward_fn(tmp_pred_original_sample_decoded).to(torch.float32) # type: ignore
                        tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32)
                        
                        # Calculate approximate guidance noise for maximizing reward
                        tmp_approx_guidance = torch.autograd.grad(
                            outputs=tmp_log_twist_func, 
                            inputs=latents_one_hot,
                            grad_outputs=torch.ones_like(tmp_log_twist_func)
                        )[0].detach()[..., :self.codebook_size] # type: ignore
                        # tmp_approx_guidance = torch.zeros_like(latents_one_hot[..., :self.transformer.config.codebook_size]) # type: ignore
                        
                        logits[batch_size*idx : batch_size*(idx+1)] = tmp_logits.detach().clone()
                        rewards[batch_size*idx : batch_size*(idx+1)] = tmp_rewards.detach().clone()
                        log_twist_func[batch_size*idx : batch_size*(idx+1)] = tmp_log_twist_func.detach().clone()
                        approx_guidance[batch_size*idx : batch_size*(idx+1)] = tmp_approx_guidance.clone()
                
                if torch.isnan(log_twist_func).any():
                    if verbose:
                        print("NaN in log twist func, changing it to 0")
                    log_twist_func[:] = torch.nan_to_num(log_twist_func)
                if torch.isnan(approx_guidance).any():
                    if verbose:
                        print("NaN in approx guidance, changing it to 0")
                    approx_guidance[:] = torch.nan_to_num(approx_guidance)
            
            else:
                for idx in range(math.ceil(num_particles / batch_size)):
                    batch_latents = latents[batch_size*idx : batch_size*(idx+1)].clone()
                    tmp_logits = self.get_unconditional_logits(batch_latents)
                        
                    logits[batch_size*idx : batch_size*(idx+1)] = tmp_logits.detach().clone()
            
            if verbose:
                print("Expected rewards of proposals: ", rewards)
                print("Approx guidance: ", approx_guidance.flatten(start_dim=1).sum(dim=1))
                print("Approx guidance norm: ", (approx_guidance ** 2).mean().sqrt())
        
        
        #5. Inference steps
        bar = range(num_inference_steps) if disable_progress_bar else tqdm(range(num_inference_steps), leave=False)
        for i in bar:
            assert latents is not None
            if verbose:
                print("\n", "-"*50, i, "-"*50, "\n")
                    
            log_twist_func_prev = log_twist_func.clone() # Used to calculate weight later
            
            _calc_guidance()
            
            with torch.no_grad():
                if i >= start:
                    ################### Select Temperature ###################
                    if isinstance(tempering_schedule, float) or isinstance(tempering_schedule, int):
                        min_scale = min((tempering_gamma * (i - start))**tempering_schedule, 1.)
                        min_scale_next = min(tempering_gamma * (i + 1 - start), 1.)
                    elif tempering_schedule == "exp":
                        min_scale = min((1 + tempering_gamma) ** (i - start) - 1, 1.)
                        min_scale_next = min((1 + tempering_gamma) ** (i + 1 - start) - 1, 1.)
                    elif tempering_schedule == "adaptive":
                        min_scale = scale_factor
                    else:
                        min_scale = 1.
                        min_scale_next = 1.
                    
                    if tempering == "adaptive" and i > 0 and min_scale < 1.:
                        scale_factor = adaptive_tempering(
                            log_w.view(num_particles, -1).T, 
                            log_prob_diffusion.view(num_particles, -1).T, 
                            log_twist_func.view(num_particles, -1).T, 
                            log_prob_proposal.view(num_particles, -1).T, 
                            log_twist_func_prev.view(num_particles, -1).T, 
                            min_scale=min_scale, ess_threshold=ess_threshold
                        )
                        min_scale_next = scale_factor.clone()
                    elif tempering == "adaptive" and i == 0:
                        pass
                    elif tempering == "schedule":
                        scale_factor = min_scale
                    else:
                        scale_factor = 1.

                    if verbose:
                        print("scale factor (lambda_t): ", scale_factor)
                        print("min scale next (lambda_t-1): ", min_scale_next)
                    
                    log_twist_func *= scale_factor
                    approx_guidance *= min_scale_next
                    
                    print("Approx guidance norm after scale: ", (approx_guidance ** 2).mean().sqrt())
                    
                    ################### Weight & Resample (Importance Sampling) ###################
                    
                    # Calculate weights for samples from proposal distribution
                    incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
                    
                    log_w += incremental_log_w.detach()

                    ess = compute_ess_from_log_w(log_w).item()

                    # resample latents and corresponding variables
                    resample_indices, is_resampled, log_w = resample_fn(log_w.view(1, num_particles))
                    assert len(log_w) == 1 and len(resample_indices) == 1
                    log_w, resample_indices = log_w[0], resample_indices[0]

                    if verbose:
                        if is_resampled.any():
                            print("\n" + "="*50)
                            print(" 🔁✨ RESAMPLED! ✨🔁 ")
                            print("="*50 + "\n")
                        print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
                        print("log_twist_func - log_twist_func_prev: ", log_twist_func - log_twist_func_prev)
                        print("Incremental weight: ", incremental_log_w)
                        print("Effective sample size: ", ess)
                        print("Resampled particles indices: ", resample_indices)
                        
                    # Update variables based on resampling
                    latents = latents[resample_indices] # type: ignore
                    logits = logits[resample_indices]
                    approx_guidance = approx_guidance[resample_indices]
                    rewards = rewards[resample_indices]
                    log_twist_func = log_twist_func[resample_indices]
                
                ################### Propose Particles ###################    
                # Sample from proposal distribution
                latents, log_prob_proposal, log_prob_diffusion = self.scheduler.sample_with_approx_guidance(
                    logits=logits,
                    approx_guidance=approx_guidance,
                    latents=latents,
                    step=i,
                )
                
        assert latents is not None
        # Weights for Final samples
        if verbose:
            print("\n", "-"*50, "final", "-"*50, "\n")
            
        log_twist_func_prev = log_twist_func.clone()
        
        #6. Decode latents to get images
        images = []
        for idx in range(math.ceil(num_particles / batch_size)):
            latents_one_hot = F.one_hot(
                latents[batch_size*idx : batch_size*(idx+1)], 
                num_classes=self.codebook_size + 1
            ).float()
            tmp_images = self.decode_one_hot_latents(latents_one_hot)
            
            # Calculate rewards
            tmp_rewards = reward_fn(tmp_images).to(torch.float32) # type: ignore
            tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32)
            
            rewards[batch_size*idx : batch_size*(idx+1)] = tmp_rewards.detach().clone()
            log_twist_func[batch_size*idx : batch_size*(idx+1)] = tmp_log_twist_func.detach().clone()
            images.append(tmp_images)
        images = torch.cat(images, dim=0)
        
        if verbose:
            print("Final rewards: ", rewards)
            
        incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
        log_w += incremental_log_w.detach()

        ess = compute_ess_from_log_w(log_w).item()

        if verbose:
            print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
            print("log_twist_func - log_twist_func_prev: ", log_twist_func - log_twist_func_prev)
            print("Incremental weight: ", incremental_log_w)
            print("Weight: ", log_w)
            print("Effective sample size: ", ess)
        
        return images
    
    def get_pred_original_sample(
        self,
        logits: torch.Tensor,
        sample_one_hot: torch.Tensor, # one_hot_sample
        use_continuous_formualtion=True,
    ) -> torch.Tensor:

        vocab_size = sample_one_hot.shape[-1]
        codebook_size, height, width = self.codebook_size, self.latent_height, self.latent_width
        batch_size = len(sample_one_hot)
        
        sample_one_hot = sample_one_hot.reshape(batch_size, height * width, vocab_size)
        logits = logits.reshape(batch_size, height * width, codebook_size)

        pred_original_sample = F.gumbel_softmax(
            logits=logits,
            hard=True,
            dim=-1,
        )
        
        if use_continuous_formualtion:
            # Carry Over Unmaksing - continuous formulation
            pred_original_sample = (
                pred_original_sample * (1 - sample_one_hot[..., :codebook_size].sum(dim=-1, keepdim=True)) +
                sample_one_hot[..., :codebook_size]
            )
        else:
            pred_original_sample = torch.where(
                sample_one_hot[..., :codebook_size].sum(dim=-1, keepdim=True) == 0,
                pred_original_sample,
                sample_one_hot[..., :codebook_size]
            )

        pred_original_sample = pred_original_sample.reshape(batch_size, height, width, codebook_size)
        return pred_original_sample
                    
    def get_unconditional_logits(self, latents):
        nb_sample = len(latents)
        drop = torch.ones(nb_sample, dtype=torch.bool).to(self._execution_device)
        dummy_labels = torch.zeros(nb_sample, dtype=torch.long, device=self._execution_device)
        with torch.autocast(device_type="cuda", enabled=self.use_mixed_precision):
            logits = self.transformer(latents, dummy_labels, drop)
        logits = logits.float()
        logits = logits.view(nb_sample, self.latent_height, self.latent_width, -1)
        return logits[..., :self.codebook_size]

    
    # def decode_latents(self, latents):
    #     images = self.vqvae.decode_code(latents)
    #     images = torch.clamp(images, -1, 1)
    #     images = (images + 1.0) / 2.0
    #     return images
    
    def decode_one_hot_latents(self, latents_one_hot):
        # get quantized latent vectors
        if self.vqvae.quantize.l2_norm:
            embedding = F.normalize(self.vqvae.quantize.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.vqvae.quantize.embedding.weight
        z_q = latents_one_hot[..., :self.codebook_size] @ embedding
        z_q = z_q.reshape(-1, self.latent_height,  self.latent_width, embedding.size(1))
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
            
        images = self.vqvae.decode(z_q)
        images = torch.clamp(images, -1, 1)
        images = (images + 1.0) / 2.0
        return images
