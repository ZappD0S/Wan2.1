import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce

from ..modules.model import rope_apply


def compute_attn_weights(query, key, chunk_size=512):
    N, L, _, E = query.shape
    _, S, _, _ = key.shape
    scale_factor = 1.0 / math.sqrt(E)

    avg_attn_weights = query.new_empty(N, L, S)
    for i in range(0, L, chunk_size):
        start = i
        end = min(i + chunk_size, L)
        query_chunk = query[:, start:end, :, :]

        attn_scores_chunk = einsum(query_chunk, key, "N L H E, N S H E -> N L H S")
        attn_weights_chunk = F.softmax(scale_factor * attn_scores_chunk, dim=-1)
        avg_attn_weights[:, start:end, :] = reduce(
            attn_weights_chunk, "N L H S -> N L S", reduction="mean"
        )

    return avg_attn_weights


def compute_spatial_bias(H, W, T, face_masks, device):
    y_coords = torch.arange(H, device=device).float()
    x_coords = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    flat_y = grid_y.flatten()  # [H*W]
    flat_x = grid_x.flatten()  # [H*W]

    biases = []

    # compute the spatial map for each character
    for mask in face_masks:
        mask_2d = mask.view(H, W)

        if mask_2d.sum() > 0:
            # find centroid in frame 0
            indices = torch.nonzero(mask_2d).float()
            center_y = indices[:, 0].mean()
            center_x = indices[:, 1].mean()

            # euclidean distance squared
            dist_sq = (flat_y - center_y) ** 2 + (flat_x - center_x) ** 2

            # normalize by max possible distance to keep weight scale consistent
            normalized_dist = dist_sq / (H**2 + W**2)
            biases.append(normalized_dist)
        else:
            biases.append(torch.zeros_like(flat_y))

    # stack: [Num_Chars, H*W]
    spatial_map = torch.stack(biases, dim=0)

    # repeat across all T frames: [Num_Chars, T*H*W]
    video_map = spatial_map.repeat(1, T)

    # add batch dimension: [1, Num_Chars, L_video]
    return video_map.unsqueeze(0)


def compute_simil_masks(
    query,
    key,
    face_masks,
    grid_sizes,
    use_rope=False,
    freqs=None,
    chunk_size=512,
    spatial_weight=0.0,
):
    def _weighted_average(x, weights, dim):
        weights = weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-6)
        return (x * weights).sum(dim=dim)

    [[T, H, W]] = grid_sizes

    key_first_frame = rearrange(
        key, "1 (T H W) num_heads E -> 1 T (H W) num_heads E", T=T, H=H, W=W
    )[:, 0]

    if use_rope:
        if freqs is None:
            raise ValueError

        c = freqs.shape[1]
        time_dim = c - 2 * (c // 3)

        tracking_freqs = freqs.clone()
        # We take the 'Index 0' (no rotation) values from the Time columns
        # and expand them to all T indices.
        # This makes every frame "look like" Frame 0 to the temporal encoder.
        tracking_freqs[:, :time_dim] = tracking_freqs[0, :time_dim]

        # Query will now have correct Spatial RoPE but NO Temporal Drift
        query = rope_apply(query, grid_sizes, tracking_freqs)

        key_first_frame = rope_apply(
            key_first_frame, torch.tensor([[1, H, W]]), tracking_freqs
        )

    attn_weights = compute_attn_weights(query, key_first_frame, chunk_size=chunk_size)

    background_mask = ~(face_masks.any(dim=0, keepdim=True))
    all_masks = torch.cat([face_masks, background_mask])

    simil_scores_list = []
    for target_mask in all_masks:
        target_mask = rearrange(target_mask, "S -> 1 1 S")
        simil_scores_list.append(
            _weighted_average(attn_weights, weights=target_mask.float(), dim=2)
        )

    # Shape: [N, L, Num_Concepts + 1]
    simil_scores = torch.stack(simil_scores_list, dim=1)

    # We apply the bias ONLY to the characters, not the background (last index).
    # Background usually spans the whole screen, so centroids make no sense for it.

    if spatial_weight > 0:
        # Calculate distance penalty for characters
        dist_penalty = compute_spatial_bias(H, W, T, face_masks, query.device)

        # Subtract the penalty from the similarity score
        # "High Distance" = "Lower Score"
        # We perform the operation only on the character slots (:-1)
        simil_scores[:, :-1, :] -= dist_penalty * spatial_weight

    # 4. Winner Takes All
    simil_masks = simil_scores == simil_scores.max(dim=1, keepdim=True).values

    # Remove background
    simil_masks = simil_masks[:, :-1, :]

    return simil_masks
