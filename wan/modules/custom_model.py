# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat

from ..utils.simil_mask import compute_simil_masks
from .attention import flash_attention
from .model import (
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

__all__ = ["CustomWanModel"]

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


class CustomWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, bias_kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (T, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        x = flash_attention(
            q=q, k=k, v=v, k_lens=seq_lens, window_size=self.window_size
        )

        # output
        x = x.flatten(2)

        face_masks = bias_kwargs["face_masks"]

        T, H, W = grid_sizes[0]
        k_first_frame = rearrange(
            k, "1 (T H W) num_heads E -> 1 T (H W) num_heads E", T=T, H=H, W=W
        )[:, 0]

        # these masks say, for every frame, where the faces of each person is supposed to be
        simil_masks = compute_simil_masks(q, k_first_frame, face_masks)

        x = self.o(x)
        return x, simil_masks


class CustomWanI2VCrossAttention(CustomWanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, grid_sizes, bias_kwargs):  # type: ignore
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """

        def _compute_attn_mask(
            main_token_mask, token_data_list, simil_masks, wlw_matrix
        ):
            T, H, W = grid_sizes[0]
            L = T * H * W
            attn_mask = repeat(
                torch.ones_like(main_token_mask), "S -> 1 L S", L=L
            ).contiguous()

            for tokens_data in token_data_list:
                inds = tokens_data["inds"]
                char_descr_token_masks = tokens_data["char_descr_token_masks"]
                action_token_mask = tokens_data["action_token_mask"]
                action_token_mask = rearrange(action_token_mask, "S -> 1 1 S")

                single_ind = len(inds) == 1

                time_mask = wlw_matrix[inds * 2 if single_ind else inds]
                time_mask = repeat(time_mask, "T -> 1 (T H W) 1", H=H, W=W)
                attn_mask = torch.where(
                    action_token_mask, attn_mask & time_mask, attn_mask
                )

                # For now a single index means no interaction.
                # It's just for a piece of text that concern a single character.
                if single_ind:
                    (i,) = inds
                    face_mask = simil_masks[:, i]
                    face_mask = rearrange(face_mask, "N L -> N L 1")
                    attn_mask = torch.where(
                        action_token_mask, attn_mask & face_mask, attn_mask
                    )
                    continue

                if not char_descr_token_masks:
                    raise ValueError

                for i, tokens_mask in zip(inds, char_descr_token_masks):
                    face_mask = simil_masks[:, i]
                    tokens_mask = rearrange(tokens_mask, "S -> 1 1 S")
                    face_mask = rearrange(face_mask, "N L -> N L 1")
                    attn_mask = torch.where(
                        tokens_mask, attn_mask & face_mask, attn_mask
                    )

            return attn_mask

        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # this merges heads and channel dims
        img_x = img_x.flatten(2)

        bias_method = bias_kwargs["bias_method"]

        if not bias_kwargs["bias"] or bias_method == "none":
            x = flash_attention(q, k, v, k_lens=None)
            # this merges heads and channel dims
            x = x.flatten(2)

            x = self.o(x + img_x)
            return x

        tokens_data_list = bias_kwargs["tokens_data_list"]
        simil_masks = bias_kwargs["simil_masks"]
        wlw_matrix = bias_kwargs["wlw_matrix"]
        full_token_mask = bias_kwargs["full_token_mask"]

        attn_mask = _compute_attn_mask(
            main_token_mask=full_token_mask,
            token_data_list=tokens_data_list,
            simil_masks=simil_masks,
            wlw_matrix=wlw_matrix,
        )

        if bias_method == "regional_prompting":
            x = flash_attention(q, k, v, k_lens=None)
            # this merges heads and channel dims
            x = x.flatten(2)

            # the head before seq_len is necessary for scaled_dot_product_attention
            q = rearrange(q, "N L H E -> N H L E")
            k = rearrange(k, "N S H E -> N H S E")
            v = rearrange(v, "N S H E -> N H S E")

            assert attn_mask.dtype == torch.bool
            y = F.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=attn_mask
            )
            # merge heads and channel dims
            y = rearrange(y, "N H L E -> N L (H E)")

            beta = bias_kwargs["beta"]
            assert (0.0 <= beta) and (beta <= 1.0)
            # NOTE: since self.o is just a linear (no activation function) it doesn't matter if
            # we apply before or after computing this sum
            x = (1 - beta) * x + beta * y

        elif bias_method == "ediff-i":
            # the head before seq_len is necessary for scaled_dot_product_attention
            q = rearrange(q, "N L H E -> N H L E")
            k = rearrange(k, "N S H E -> N H S E")
            v = rearrange(v, "N S H E -> N H S E")

            norm_t = bias_kwargs["normalized_timestep"]
            strength = bias_kwargs["strength"]
            attn_mask = strength * norm_t * attn_mask.float()
            x = F.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=attn_mask
            )

            # merge heads and channel dims
            x = rearrange(x, "N H L E -> N L (H E)")
        else:
            raise ValueError("Unknown bias method!")

        x = self.o(x + img_x)

        return x


WAN_CROSSATTENTION_CLASSES = {
    "i2v_cross_attn": CustomWanI2VCrossAttention,
}


class CustomWanAttentionBlock(nn.Module):
    _keep_in_fp32_modules = ["norm1", "norm2", "norm3", "modulation"]

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CustomWanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, bias_kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (T, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention

        y, simil_masks = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs,
            bias_kwargs=bias_kwargs,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn

        bias_kwargs = bias_kwargs.copy()
        bias_kwargs["simil_masks"] = simil_masks
        x_cross_attn = self.cross_attn(
            self.norm3(x), context, grid_sizes, bias_kwargs=bias_kwargs
        )
        x = x + x_cross_attn
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])

        with torch.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5]

        return x, simil_masks


class Head(nn.Module):
    _keep_in_fp32_modules = ["norm", "head", "modulation"]

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280)
            )

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class CustomWanModel(ModelMixin, ConfigMixin):
    _keep_in_fp32_modules = ["time_embedding", "time_projection"]

    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v", "vace"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CustomWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

    def forward(self, x, t, context, seq_len, bias_kwargs, clip_fea=None, y=None):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, T, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, T, H / 8, W / 8]
        """
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        bias_kwargs = bias_kwargs.copy()
        T, H, W = [int(s) for s in grid_sizes[0]]

        face_masks = bias_kwargs["face_masks"]
        face_masks = rearrange(face_masks.float(), "N H W -> N 1 H W")
        face_masks = F.interpolate(face_masks, size=(H, W), mode='nearest')
        face_masks = rearrange(face_masks.bool(), "N 1 H W -> N (H W)")
        bias_kwargs["face_masks"] = face_masks.to(device)

        wlw_matrix = bias_kwargs["wlw_matrix"]

        N1, N2, _ = wlw_matrix.shape
        assert N1 == N2
        wlw_matrix = rearrange(wlw_matrix.float(), "N1 N2 T -> (N1 N2) 1 T")
        wlw_matrix = F.interpolate(wlw_matrix, size=(T,), mode='nearest')
        wlw_matrix = rearrange(
            wlw_matrix.bool(), "(N1 N2) 1 T -> N1 N2 T", N1=N1, N2=N2
        )
        bias_kwargs["wlw_matrix"] = wlw_matrix.to(device)

        # time embeddings
        with torch.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        def pad_to_length(x, dim=-1):
            pad_len = self.text_len - x.size(dim)
            assert pad_len >= 0

            offset = abs(dim) if dim < 0 else x.ndim - dim
            padding = [0] * (2 * offset)
            padding[-1] = pad_len
            return F.pad(x, padding)

        context = self.text_embedding(
            torch.stack([pad_to_length(u, dim=-2) for u in context])
        )

        full_token_mask = bias_kwargs["full_token_mask"]
        full_token_mask = pad_to_length(full_token_mask)
        bias_kwargs["full_token_mask"] = full_token_mask

        tokens_data_list = []
        for tokens_data in bias_kwargs["tokens_data_list"]:
            td = tokens_data.copy()
            td["action_token_mask"] = pad_to_length(td["action_token_mask"])
            td["char_descr_token_masks"] = [
                pad_to_length(mask) for mask in td["char_descr_token_masks"]
            ]
            tokens_data_list.append(td)

        bias_kwargs["tokens_data_list"] = tokens_data_list

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        shared_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
        )

        bias = bias_kwargs.pop("bias")
        blocks_bias_schedule = bias_kwargs.pop("blocks_bias_schedule")

        simil_masks_list = []

        for i, block in enumerate(self.blocks):
            bias_block = bias and blocks_bias_schedule[i]

            shared_kwargs["bias_kwargs"] = bias_kwargs | {"bias": bias_block}
            x, simil_masks = block(x, **shared_kwargs)

            simil_masks_list.append(simil_masks)

            if not bias_block:
                continue

        simil_masks = torch.stack(simil_masks_list, dim=1)
        simil_masks = rearrange(simil_masks, "... (T H W) -> ... T H W", T=T, H=H, W=W)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x], simil_masks

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, T, H / 8, W / 8]
        """

        c = self.out_dim
        p, q, r = self.patch_size
        out = []

        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)]
            f, h, w = v
            u = rearrange(
                u,
                "(f h w) (p q r c) -> c (f p) (h q) (w r)",
                f=f,
                h=h,
                w=w,
                p=p,
                q=q,
                r=r,
                c=c,
            )
            out.append(u)

        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
