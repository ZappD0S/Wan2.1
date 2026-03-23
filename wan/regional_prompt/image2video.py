# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from typing import Any, TypedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from ..distributed.fsdp import shard_model
from ..modules.clip import CLIPModel
from ..modules.custom_model import CustomWanModel
from ..modules.t5 import T5EncoderModel
from ..modules.vae import WanVAE
from ..utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.subsequence import get_nested_subsequence_mask

NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


class WanI2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),  # pyright: ignore[reportArgumentType]
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,  # pyright: ignore[reportArgumentType]
        )

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer),
        )

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model: CustomWanModel
        self.model = CustomWanModel.from_pretrained(
            checkpoint_dir, torch_dtype=config.param_dtype
        )

        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import (  # type: ignore
                get_sequence_parallel_world_size,
            )

            from .distributed.xdit_context_parallel import (  # type: ignore
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(  # ty:ignore[invalid-assignment]
                    usp_attn_forward, block.self_attn
                )
            self.model.forward = types.MethodType(usp_dit_forward, self.model)  # ty:ignore[invalid-assignment]
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)  # ty:ignore[invalid-argument-type]

        self.sample_neg_prompt = config.sample_neg_prompt

    def _img_to_tensor(self, img: Image.Image) -> torch.Tensor:
        return TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    def _get_lat_h_w(self, size: tuple[int, int], max_area: float):
        h, w = size
        aspect_ratio = h / w
        assert aspect_ratio <= 1

        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )
        return lat_h, lat_w

    def _generate_noise(
        self,
        size: tuple[int, int],
        max_area: float,
        frame_num: int,
        seed_g: torch.Generator,
    ):
        lat_h, lat_w = self._get_lat_h_w(size, max_area)

        r = self.vae_stride[0]
        noise = torch.randn(
            16,
            (frame_num + r - 1) // r,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )
        return noise

    def _build_latents(
        self,
        img_tensor: torch.Tensor,
        frame_num: int,
        max_area,
    ):
        _, h, w = img_tensor.shape
        lat_h, lat_w = self._get_lat_h_w((h, w), max_area)

        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        r = self.vae_stride[0]
        max_seq_len = (
            ((frame_num - 1) // r + 1)
            * lat_h
            * lat_w
            // (self.patch_size[1] * self.patch_size[2])
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        padding_frames = torch.zeros(3, frame_num - 1, h, w)
        resized_img = F.interpolate(
            img_tensor.unsqueeze(0).cpu(), size=(h, w), mode="bicubic"
        )
        resized_img = rearrange(resized_img, "1 C H W -> C 1 H W")

        input_sequence = torch.concat([resized_img, padding_frames], dim=1)
        input_sequence = input_sequence.to(self.device)

        y = self.vae.encode([input_sequence])[0]
        msk = y.new_zeros(r, (frame_num + r - 1) // r, lat_h, lat_w)
        # set the first frame (in latent space) to one
        msk[:, 0] = 1
        y = torch.concat([msk, y])

        return y, max_seq_len

    def _build_context(
        self,
        img_tensor: torch.Tensor,
        prompt_sentences: list[str],
        n_prompt: str,
        bias_kwargs,
        offload_model: bool,
    ):
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if self.t5_cpu:
            t5_device = torch.device('cpu')
        else:
            self.text_encoder.model.to(self.device)
            t5_device = self.device

        self.clip.model.to(self.device)

        tokenizer = self.text_encoder.tokenizer
        context_null = self.text_encoder([n_prompt], t5_device)

        sentence_contexts = []
        single_char_img_contexts = []

        full_token_masks_list = []
        tokens_data_list = []

        bias_kwargs = bias_kwargs.copy()
        prompt_data_list = bias_kwargs.pop("prompt_data_list")
        general_prompt = bias_kwargs.pop("general_prompt")

        if general_prompt is not None:
            general_prompt_context = self.text_encoder([general_prompt], t5_device)
        else:
            general_prompt_context = []

        for sentence, prompt_data in zip(
            prompt_sentences, prompt_data_list, strict=True
        ):
            control_prompts = prompt_data["control_prompts"]
            [sentence_context] = self.text_encoder([sentence], t5_device)
            single_char_img = prompt_data["single_char_img"]

            if single_char_img is not None:
                single_char_img_tensor = self._img_to_tensor(single_char_img)
                [single_char_img_context] = self.clip.visual(
                    [single_char_img_tensor[:, None, :, :]]
                )
                single_char_img_contexts.append(single_char_img_context)

            cum_len = sum(ctx.size(0) for ctx in sentence_contexts)

            [full_token_ids], [full_token_mask_np] = tokenizer(
                sentence,
                return_mask=True,
                padding=False,
                add_special_tokens=True,
                return_tensors="np",
            )
            full_token_mask = torch.from_numpy(full_token_mask_np).to(
                dtype=torch.bool, device=self.device
            )

            for inds, prompt_data in control_prompts:
                add_special_tokens = sentence.endswith(prompt_data["prompt"])
                [action_token_ids] = tokenizer(
                    prompt_data["prompt"],
                    padding=False,
                    add_special_tokens=add_special_tokens,
                    return_tensors="np",
                )

                action_token_mask_np = get_nested_subsequence_mask(
                    full_token_ids, [action_token_ids]
                )
                action_token_mask_np = np.append(
                    np.zeros(cum_len, dtype=bool), action_token_mask_np
                )
                action_token_mask = torch.from_numpy(action_token_mask_np).to(
                    dtype=torch.bool, device=self.device
                )

                char_descr_masks_list = []
                for char_descr in prompt_data["char_descr_list"]:
                    [char_descr_token_ids] = tokenizer(
                        char_descr,
                        padding=False,
                        add_special_tokens=False,
                        return_tensors="np",
                    )

                    char_descr_mask_np = get_nested_subsequence_mask(
                        full_token_ids,
                        [action_token_ids, char_descr_token_ids],
                    )
                    char_descr_mask_np = np.append(
                        np.zeros(cum_len, dtype=bool), char_descr_mask_np
                    )
                    char_descr_mask = torch.from_numpy(char_descr_mask_np).to(
                        dtype=torch.bool, device=self.device
                    )

                    assert not (char_descr_mask & (~action_token_mask)).any()
                    char_descr_masks_list.append(char_descr_mask)

                tokens_data_list.append(
                    {
                        "inds": inds,
                        "action_token_mask": action_token_mask,
                        "char_descr_token_masks": char_descr_masks_list,
                    }
                )

            sentence_contexts.append(sentence_context)
            full_token_masks_list.append(full_token_mask)

        context = [torch.cat(sentence_contexts)]
        full_token_mask = torch.cat(full_token_masks_list)

        if self.t5_cpu:
            general_prompt_context = [t.to(self.device) for t in general_prompt_context]
            sentence_contexts = [t.to(self.device) for t in sentence_contexts]
            single_char_img_contexts = [
                t.to(self.device) for t in single_char_img_contexts
            ]
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        elif offload_model:
            self.text_encoder.model.cpu()

        clip_context = self.clip.visual([img_tensor[:, None, :, :]])

        if offload_model:
            self.clip.model.cpu()
            torch.cuda.empty_cache()

        assert len(single_char_img_contexts) == 0 or len(
            single_char_img_contexts
        ) == len(sentence_contexts)
        bias_kwargs["general_prompt_context"] = general_prompt_context
        bias_kwargs["sentence_contexts"] = sentence_contexts
        bias_kwargs["single_char_img_contexts"] = single_char_img_contexts
        bias_kwargs["tokens_data_list"] = tokens_data_list
        bias_kwargs["full_token_mask"] = full_token_mask

        return context, context_null, clip_context, bias_kwargs

    def _get_scheduler(self, sample_solver, sampling_steps, shift):
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift
            )
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=self.device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError("Unsupported solver.")

        return sample_scheduler, timesteps

    def _compute_noise_pred(
        self,
        latent,
        timestep,
        y,
        context,
        context_null,
        clip_context,
        max_seq_len,
        guide_scale,
        offload_model,
        bias_kwargs,
    ):
        shared_kwargs = {
            "t": timestep,
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": y,
        }

        cond_kwargs = shared_kwargs.copy()
        cond_kwargs["context"] = context
        cond_kwargs["bias_kwargs"] = bias_kwargs

        [noise_pred_cond], simil_masks = self.model(latent, **cond_kwargs)
        assert "simil_masks" not in bias_kwargs

        if offload_model:
            noise_pred_cond = noise_pred_cond.to('cpu')
            simil_masks = simil_masks.to('cpu')

            torch.cuda.empty_cache()

        uncond_kwargs = shared_kwargs.copy()
        uncond_kwargs["context"] = context_null
        uncond_kwargs["bias_kwargs"] = bias_kwargs | {"bias": False}

        [noise_pred_uncond], *_ = self.model(latent, **uncond_kwargs)

        if offload_model:
            noise_pred_uncond = noise_pred_uncond.to('cpu')
            torch.cuda.empty_cache()

        noise_pred = noise_pred_uncond + guide_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        return noise_pred, simil_masks

    def generate_from_latents(
        self,
        img: Image.Image,
        prompt_sentences: list[str],
        bias_kwargs,
        latent,
        t_i=None,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt=NEGATIVE_PROMPT,
        seed: int | torch.Generator = -1,
        offload_model=True,
    ):
        sample_scheduler, timesteps = self._get_scheduler(
            sample_solver, sampling_steps, shift
        )

        if t_i is None:
            i0 = 0
        else:
            mask = timesteps == t_i

            if not mask.any():
                raise ValueError

            idx = mask.nonzero(as_tuple=True)

            if len(idx[0]) != 1:
                raise ValueError(f"Expected exactly one match, found {len(idx[0])}")

            i0 = idx[0][0].item()

        if isinstance(seed, int):
            seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            seed_g = seed
        else:
            raise ValueError

        timestep_bias_schedule = bias_kwargs.pop("timestep_bias_schedule")

        bias_kwargs = bias_kwargs.copy()
        img_tensor = self._img_to_tensor(img)
        y, max_seq_len = self._build_latents(img_tensor, frame_num, max_area)
        context, context_null, clip_context, bias_kwargs = self._build_context(
            img_tensor,
            prompt_sentences,
            n_prompt,
            bias_kwargs=bias_kwargs,
            offload_model=offload_model,
        )

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        temp_x0 = None

        with (
            torch.autocast("cuda", dtype=self.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)  # ty:ignore[invalid-argument-type]
            simil_masks_list = []
            for i, t in enumerate(tqdm(timesteps[i0:])):
                bias_timestep = timestep_bias_schedule[i + i0]

                timestep = torch.tensor([t], device=self.device)
                latent = latent.to(self.device)

                norm_t = timestep / self.num_train_timesteps
                assert (0.0 <= norm_t) and (norm_t <= 1.0)
                noise_pred, simil_masks = self._compute_noise_pred(
                    [latent],
                    timestep,
                    [y],
                    context=context,
                    context_null=context_null,
                    clip_context=clip_context,
                    bias_kwargs=bias_kwargs
                    | {"normalized_timestep": norm_t, "bias": bias_timestep},
                    max_seq_len=max_seq_len,
                    guide_scale=guide_scale,
                    offload_model=offload_model,
                )

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device
                )

                simil_masks = simil_masks.to("cpu")
                simil_masks_list.append(simil_masks)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latent = temp_x0.squeeze(0)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            videos = [None]
            if self.rank == 0:
                videos = self.vae.decode([latent.to(self.device)])

        del latent, temp_x0

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        if dist.is_initialized():
            dist.barrier()

        simil_masks = torch.stack(simil_masks_list, dim=1)

        extra_data = {"simil_masks": simil_masks}
        return (videos[0], extra_data) if self.rank == 0 else None

    def generate(
        self,
        prompt_sentences: list[str],
        img: Image.Image,
        bias_kwargs,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt=NEGATIVE_PROMPT,
        seed=-1,
        offload_model=True,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        w, h = img.size
        noise = self._generate_noise((h, w), max_area, frame_num, seed_g)

        return self.generate_from_latents(
            img,
            prompt_sentences,
            bias_kwargs,
            noise,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed_g,
            offload_model=offload_model,
        )
