from typing import Optional, Union, Callable

import math
import torch
from tqdm import tqdm
import torch.nn.functional as F

from smc_utils import compute_ess_from_log_w, resampling_function, adaptive_tempering
from plot_utils import show_images_grid, plot_histogram, plot_errors


def assert_one_hot(x):
    assert ((x == 0) | (x == 1)).all() and (x.sum(dim=-1) == 1).all(), "Tensor is not one-hot"

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of log(1 - exp(x)) for x < 0.
    """
    return torch.where(
        x > -1,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )

def logmeanexp(x, dim=None, keepdim=False):
    """Numerically stable log-mean-exp using torch.logsumexp."""
    if dim is None:
        x = x.view(-1)
        dim = 0
    # log-sum-exp with or without keeping the reduced dim
    lse = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    # subtract log(N) to convert sum into mean (broadcasts correctly)
    return lse - math.log(x.size(dim))


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

    def sample_with_approx_guidance(
        self,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
        latents: torch.Tensor,
        step: int,
    ):
        B, H, W, C = logits.shape
        assert latents.shape == (B, H, W)
        self.mask_token_id = C-1
        
        logits = logits.reshape(B, H*W, C)
        approx_guidance = approx_guidance.reshape(B, H*W, C)
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
        
        proposal_logits = (p_theta_logits + approx_guidance).log_softmax(dim=-1)
        assert torch.allclose(torch.exp(proposal_logits).sum(dim=-1), torch.ones(B, H*W, device=logits.device))
        
        # modify proposal logits to have the same mask schedule as the original logits
        proposal_logits[..., :self.mask_token_id] += (
            torch.logsumexp(p_theta_logits[..., :self.mask_token_id], dim=(1, 2), keepdim=True) - 
            torch.logsumexp(proposal_logits[..., :self.mask_token_id], dim=(1, 2), keepdim=True)
        )
        proposal_logits[..., :self.mask_token_id] = torch.where(
            proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1, keepdim=True) >= 0,
            proposal_logits[..., :self.mask_token_id].log_softmax(dim=-1),
            proposal_logits[..., :self.mask_token_id]
        )
        assert not (proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1) > 1e-6).any(), proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1).max()
        proposal_logits[..., self.mask_token_id] = (
            log1mexp(proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1).clamp_max(0))
        )
        assert torch.allclose(torch.exp(proposal_logits).sum(dim=-1), torch.ones(B, H*W, device=logits.device)), (torch.exp(proposal_logits).sum(dim=-1) - torch.ones(B, H*W, device=logits.device)).abs().max()
        # modify proposal logits to have the same mask schedule as the original logits
        
        proposal_dist = torch.distributions.Categorical(logits=proposal_logits)
        diffusion_dist = torch.distributions.Categorical(logits=p_theta_logits)
        
        n_samples = 10000
        latent_samples = diffusion_dist.sample((n_samples,)) # shape (n_samples, B, L) 
        log_probs_proposal = proposal_dist.log_prob(latent_samples).sum(dim=2).T # shape (B, n_samples)
        log_probs_diffusion = diffusion_dist.log_prob(latent_samples).sum(dim=2).T # shape (B, n_samples)
        
        _, max_indices = (log_probs_proposal + log_probs_diffusion).max(dim=1)
        
        new_latents = latent_samples[max_indices, torch.arange(B)]
        log_prob_proposal = proposal_dist.log_prob(new_latents).sum(dim=1)
        log_prob_diffusion = diffusion_dist.log_prob(new_latents).sum(dim=1)
        
        new_latents = new_latents.reshape(B, H, W)
        
        print("Unmasked:", (new_latents != C-1).sum(dim=(1, 2)))
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
        
        # Intialize variables for SMC sampler
        logits = torch.zeros((*latents.shape, self.codebook_size + 1), device=self._execution_device) # type: ignore
        approx_guidance = torch.zeros((*latents.shape, self.codebook_size + 1), device=self._execution_device) # type: ignore 
        unscaled_approx_guidance = torch.zeros((*latents.shape, self.codebook_size + 1), device=self._execution_device) # type: ignore 
        log_w = torch.zeros(latents.shape[0], device=self._execution_device)
        log_prob_diffusion = torch.zeros(latents.shape[0], device=self._execution_device)
        log_prob_proposal = torch.zeros(latents.shape[0], device=self._execution_device)
        log_twist_func = torch.zeros(latents.shape[0], device=self._execution_device)
        log_twist_func_prev = torch.zeros(latents.shape[0], device=self._execution_device)
        rewards = torch.zeros(latents.shape[0], device=self._execution_device)
        resample_fn = resampling_function(resample_strategy=resample_strategy, ess_threshold=ess_threshold)
        scale_factor = 0.
        min_scale_next = 0.
        prev_latents = latents.clone()
        approximation_errors = []
        rewards_trace = []
        
        kl_coeff = torch.tensor(kl_coeff, device=self._execution_device).to(torch.float32) # type: ignore
        lookforward_fn = lambda r: r / kl_coeff
        
        start = int(num_inference_steps * tempering_start)
        
        def _calc_guidance():
            assert latents is not None
            predicted_rewards = (
                rewards + (
                    (F.one_hot(latents, num_classes=self.codebook_size + 1) - F.one_hot(prev_latents, num_classes=self.codebook_size + 1))
                    * unscaled_approx_guidance
                ).flatten(start_dim=1).sum(dim=1)
            )
            if (i >= start):
                imgs = []
                for idx in range(math.ceil(num_particles / batch_size)):
                    with torch.enable_grad():
                        latents_one_hot = F.one_hot(
                            latents[batch_size*idx : batch_size*(idx+1)], 
                            num_classes=self.codebook_size + 1 # type: ignore
                        ).float().requires_grad_(True)

                        tmp_logits = self.get_unconditional_logits(latents_one_hot)
                        
                        M = 1
                        tmp_rewards = torch.zeros_like(rewards[batch_size*idx : batch_size*(idx+1)]).unsqueeze(1).repeat(1, M)

                        for m_i in range(M):
                            tmp_pred_original_sample: torch.Tensor = self.get_pred_original_sample(
                                logits=tmp_logits,
                                sample_one_hot=latents_one_hot,
                                use_continuous_formualtion=True,
                            )
                            
                            tmp_pred_original_sample_decoded = self.decode_one_hot_latents(tmp_pred_original_sample)
                            
                            # Calculate rewards
                            tmp_rewards[:, m_i] = reward_fn(tmp_pred_original_sample_decoded).to(torch.float32) # type: ignore
                            
                        imgs.append(tmp_pred_original_sample_decoded.detach().cpu())
                        
                        tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32)
                        
                        tmp_log_twist_func = logmeanexp(tmp_log_twist_func, dim=1)
                        tmp_rewards = tmp_log_twist_func * kl_coeff
                        
                        # Calculate approximate guidance noise for maximizing reward
                        tmp_approx_guidance = torch.autograd.grad(
                            outputs=tmp_log_twist_func,
                            inputs=latents_one_hot,
                            grad_outputs=torch.ones_like(tmp_log_twist_func)
                        )[0].detach() # type: ignore
                        
                        logits[batch_size*idx : batch_size*(idx+1)] = tmp_logits.detach().clone()
                        rewards[batch_size*idx : batch_size*(idx+1)] = tmp_rewards.detach().clone()
                        log_twist_func[batch_size*idx : batch_size*(idx+1)] = tmp_log_twist_func.detach().clone()
                        approx_guidance[batch_size*idx : batch_size*(idx+1)] = tmp_approx_guidance.clone()
                
                show_images_grid(torch.cat(imgs, dim=0), save_file=f"output_SMC.png")
                if i%5 == 0:
                    plot_histogram(approx_guidance.cpu(), save_file=f"approx_guidance_{i}.png")
                # approx_guidance.clamp_max_(5.0)
                
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
                    
            unscaled_approx_guidance[:] = approx_guidance.clone() * kl_coeff
            rewards_trace.append(rewards.mean().cpu())
            approximation_errors.append((rewards - predicted_rewards).mean().cpu())
            
            if verbose:
                print("Expected rewards of proposals: ", rewards)
                print("Predicted rewards based on gradients: ", predicted_rewards)
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
                    
                    if verbose:
                        print("Approx guidance norm after scale: ", (approx_guidance ** 2).mean().sqrt())
                    
                    ################### Weight & Resample (Importance Sampling) ###################
                    
                    # Calculate weights for samples from proposal distribution
                    # incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
                    incremental_log_w = log_twist_func - log_twist_func_prev
                    
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
                    unscaled_approx_guidance = unscaled_approx_guidance[resample_indices]
                    rewards = rewards[resample_indices]
                    log_twist_func = log_twist_func[resample_indices]
                    
                prev_latents = latents.clone()
                
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
        
        plot_errors(approximation_errors, rewards_trace)
        
        return images, log_w
    
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
        logits = logits.reshape(batch_size, height * width, vocab_size)

        pred_sample = F.gumbel_softmax(
            logits=logits,
            hard=True,
            dim=-1,
            tau=self.scheduler.temperature,
        )
        pred_original_sample = torch.zeros_like(pred_sample)
        
        if use_continuous_formualtion:
            # Carry Over Unmaksing - continuous formulation
            pred_original_sample[..., :codebook_size] = (
                pred_sample[..., :codebook_size] * (1 - sample_one_hot[..., :codebook_size].sum(dim=-1, keepdim=True)) +
                sample_one_hot[..., :codebook_size]
            )
        else:
            pred_original_sample[..., :codebook_size] = torch.where(
                sample_one_hot[..., :codebook_size].sum(dim=-1, keepdim=True) == 0,
                pred_sample[..., :codebook_size],
                sample_one_hot[..., :codebook_size]
            )

        assert_one_hot(pred_original_sample)
        pred_original_sample = pred_original_sample.reshape(batch_size, height, width, vocab_size)
        return pred_original_sample
                    
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

