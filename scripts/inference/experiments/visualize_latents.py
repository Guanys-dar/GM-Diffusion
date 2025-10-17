# Add this code to your main script or a separate file

import torch
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import the original pipeline class
from gm_diffusion.pipelines import StableDiffusionDualUNetPipeline, retrieve_timesteps, rescale_noise_cfg

# Import necessary components if not already imported in the scope
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging, deprecate, scale_lora_layers, unscale_lora_layers

if hasattr(torch, "xla") and torch.xla.is_available():
     import torch_xla.core.xla_model as xm
     XLA_AVAILABLE = True
else:
     XLA_AVAILABLE = False

logger = logging.get_logger(__name__)



# Define the NEW subclass
class StableDiffusionDualUNetPipelineVis(StableDiffusionDualUNetPipeline):
    """
    Subclass of StableDiffusionDualUNetPipeline that adds the ability
    to return intermediate latent states for visualization.
    """
    @torch.no_grad()
    # Override the __call__ method
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # ip_adapter_image: Optional[PipelineImageInput] = None, # Corrected type hint if needed
        # ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True, # Force return_dict for intermediate handling
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        # callback_on_step_end: Optional[Union[Callable, PipelineCallback, MultiPipelineCallbacks]] = None, # Keep callback args
        # callback_on_step_end_tensor_inputs: List[str] = ["latents"], # Keep callback args
        # <<< NEW ARGUMENT >>>
        return_intermediates: bool = False,
        **kwargs,
    ):
        # --- Start of copied __call__ method ---
        # (Copy the entire __call__ method from StableDiffusionDualUNetPipeline here)
        # ... (Check inputs, define call parameters, encode prompt, prepare timesteps, etc.) ...
        # --- Modifications highlighted below ---

        # callback = kwargs.pop("callback", None)
        # callback_steps = kwargs.pop("callback_steps", None)

        # if callback is not None:
        #     deprecate(
        #         "callback",
        #         "1.0.0",
        #         "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )
        # if callback_steps is not None:
        #     deprecate(
        #         "callback_steps",
        #         "1.0.0",
        #         "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )

        if return_intermediates and not return_dict:
             logger.warning("`return_intermediates=True` requires `return_dict=True`. Forcing `return_dict=True`.")
             return_dict = True # Enforce dictionary output when intermediates are requested

        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            # callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        # (Keep original height/width logic)
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor


        # 1. Check inputs. Raise error if not correct
        callback_steps = 10
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps, # Pass callback_steps even if deprecated for original check_inputs
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,    
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        #     image_embeds = self.prepare_ip_adapter_image_embeds(
        #         ip_adapter_image,
        #         ip_adapter_image_embeds,
        #         device,
        #         batch_size * num_images_per_prompt,
        #         self.do_classifier_free_guidance,
        #     )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        # Make sure vae_scale_factor is accessible, might need self.vae_scale_factor
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        gm_latents = latents.clone() # Initial GM latents start same as SDR

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        # added_cond_kwargs = (
        #     {"image_embeds": image_embeds}
        #     # if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        #     # else None
        # )
        added_cond_kwargs = {}

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # <<< NEW: Initialize lists for intermediates >>>
        sdr_intermediate_latents = []
        gm_intermediate_latents = []

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Ensure gm_scheduler exists if used in the original loop
        import copy
        self.gm_scheduler = copy.deepcopy(self.scheduler) # Assuming original uses a separate scheduler copy

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # --- SDR Path ---
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                sdr_noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    sdr_noise_pred_uncond, sdr_noise_pred_text = sdr_noise_pred.chunk(2)
                    sdr_noise_pred = sdr_noise_pred_uncond + self.guidance_scale * (sdr_noise_pred_text - sdr_noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    sdr_noise_pred = rescale_noise_cfg(sdr_noise_pred, sdr_noise_pred_text, guidance_rescale=self.guidance_rescale)

                # Compute x0 prediction for GM UNet input
                # (Need alphas_cumprod - ensure scheduler provides this)
                if hasattr(self.scheduler, 'alphas_cumprod'):
                     alphas_cumprod = self.scheduler.alphas_cumprod.to(sdr_noise_pred.device)[t].view(-1, 1, 1, 1)
                     sqrt_alpha_cumprod = alphas_cumprod.sqrt()
                     sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod).sqrt()
                     # Use current latents before scheduler step
                     x0_latent = (latents - sqrt_one_minus_alpha_cumprod * sdr_noise_pred) / sqrt_alpha_cumprod
                else:
                     # Fallback or error if scheduler doesn't have alphas_cumprod
                     logger.warning("Scheduler lacks alphas_cumprod, cannot compute x0 for GM UNet input accurately.")
                     x0_latent = latents # Placeholder, might affect results

                # SDR Scheduler Step
                latents = self.scheduler.step(sdr_noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # --- GM Path ---
                # Scale GM latents (assuming separate scaling is needed, might depend on original logic)
                gm_latents_scaled = self.gm_scheduler.scale_model_input(gm_latents, t) # Use current gm_latents

                # Prepare GM UNet input: [x0_latent, scaled_gm_latents]
                # Ensure x0_latent is detached if necessary, though likely not needed here
                # Make sure batch sizes match if needed (x0 is single batch, gm_latents might be too)
                gm_latent_input = torch.cat([x0_latent.detach(), gm_latents_scaled], dim=1) # Use scaled GM latents

                # GM UNet prediction - NOTE: Original used prompt_embeds[1:,:,:] - check if this is correct
                # Assuming CFG for GM is not desired or handled differently based on original code structure.
                # If GM needs CFG, the input prep and model call need adjustment.
                # Let's assume the original code intends only the conditional prompt for GM:
                gm_noise_pred = self.gm_unet(
                    gm_latent_input,
                    t,
                    encoder_hidden_states=prompt_embeds[negative_prompt_embeds.shape[0]:], # Use only conditional embeds
                    timestep_cond=timestep_cond, # Check if timestep_cond needs slicing for GM
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs, # Check if added_cond_kwargs needs slicing
                    return_dict=False,
                )[0]
                # Note: No CFG guidance applied to gm_noise_pred here, assuming that matches original intent.

                # GM Scheduler Step
                gm_latents = self.gm_scheduler.step(gm_noise_pred, t, gm_latents, **extra_step_kwargs, return_dict=False)[0]


                # <<< NEW: Store intermediates >>>
                if return_intermediates:
                    sdr_intermediate_latents.append(latents.clone().cpu())
                    gm_intermediate_latents.append(gm_latents.clone().cpu())


                # --- Original Callback Logic ---
                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     # Only pass tensors listed in callback_on_step_end_tensor_inputs
                #     # Note: gm_latents cannot be passed via standard callback here
                #     for k in callback_on_step_end_tensor_inputs:
                #          if k == "latents":
                #               callback_kwargs[k] = latents # Pass the main (SDR) latents
                #          elif k == "prompt_embeds":
                #               callback_kwargs[k] = prompt_embeds
                #          elif k == "negative_prompt_embeds":
                #               callback_kwargs[k] = negative_prompt_embeds
                #          # Add other allowed tensors if necessary
                #          else:
                #               try:
                #                    callback_kwargs[k] = locals()[k]
                #               except KeyError:
                #                    logger.warning(f"Could not find tensor '{k}' for callback.")


                    # callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    # # Update variables based on callback output if needed (standard Diffusers pattern)
                    # latents = callback_outputs.pop("latents", latents)
                    # prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    # negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # Progress bar and legacy callback update (keep as is)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                        #  step_idx = i // getattr(self.scheduler, "order", 1)
                         # Legacy callback only receives main latents
                        #  callback(step_idx, t, latents)


                if XLA_AVAILABLE:
                    xm.mark_step()
        # --- End of Denoising Loop ---

        # --- Final Processing and Return ---
        # Store final latents
        sdr_latents_final = latents
        gm_latents_final = gm_latents

        # Mimic the original return structure but build a dictionary
        output = {}
        has_nsfw_concept = None # Assume no safety checker for simplicity, or handle if needed

        # Use VaeImageProcessor for postprocessing
        if not hasattr(self, "image_processor"):
             self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)


        if output_type == "latent":
            output["sdr_latent"] = sdr_latents_final
            output["gm_latent"] = gm_latents_final
            # Replicate original safety check logic if it ran on latents? Unlikely.
            output["nsfw_content_detected"] = None # Usually None for latent output
        else:
             # Decode final images using VAE
             sdr_latents_final_scaled = sdr_latents_final / self.vae.config.scaling_factor
             sdr_image_decoded = self.vae.decode(sdr_latents_final_scaled, return_dict=False, generator=generator)[0]

             gm_latents_final_scaled = gm_latents_final / self.vae.config.scaling_factor
             gm_image_decoded = self.vae.decode(gm_latents_final_scaled, return_dict=False, generator=generator)[0]

             # Run safety checker if configured and available (on decoded images)
             # This part needs careful replication if safety checker is used
             if self.safety_checker is not None and hasattr(self, "feature_extractor"):
                  # Combine images for checker or run separately? Assume separate for now.
                  sdr_image_checked, sdr_has_nsfw = self.run_safety_checker(sdr_image_decoded, device, prompt_embeds.dtype)
                  gm_image_checked, gm_has_nsfw = self.run_safety_checker(gm_image_decoded, device, prompt_embeds.dtype)
                  # Combine NSFW flags (e.g., if either is NSFW)
                  has_nsfw_concept = [s or g for s, g in zip(sdr_has_nsfw, gm_has_nsfw)]
                  # Use the checked images for postprocessing
                  sdr_image_to_process = sdr_image_checked
                  gm_image_to_process = gm_image_checked
             else:
                  has_nsfw_concept = None
                  sdr_image_to_process = sdr_image_decoded
                  gm_image_to_process = gm_image_decoded


             # Determine denormalization based on safety check
             if has_nsfw_concept is None:
                 do_denormalize = [True] * sdr_image_to_process.shape[0]
             else:
                 do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]


             # Postprocess using image_processor
             output["sdr_image"] = self.image_processor.postprocess(
                 sdr_image_to_process, output_type=output_type, do_denormalize=do_denormalize
             )
             output["gm_image"] = self.image_processor.postprocess(
                 gm_image_to_process, output_type=output_type, do_denormalize=do_denormalize
             )
             output["nsfw_content_detected"] = has_nsfw_concept


        # <<< NEW: Add intermediates to the output dictionary >>>
        if return_intermediates:
            output["sdr_intermediates"] = sdr_intermediate_latents
            output["gm_intermediates"] = gm_intermediate_latents


        # Offload models (keep original logic)
        self.maybe_free_model_hooks()

        # Return the dictionary
        # Note: We ignore the original non-return_dict path for simplicity when intermediates are requested.
        # If you absolutely need the tuple output, you'd have to reconstruct it carefully here.
        return output

# --- End of Subclass Definition ---