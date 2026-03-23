import math

import einops
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce

from ..modules.model import rope_apply


def compute_attn_weights(query, key, T=1.0, chunk_size=512):
    N, L, _, E = query.shape
    _, S, _, _ = key.shape
    scale_factor = 1.0 / math.sqrt(E)

    avg_attn_weights = query.new_empty(N, L, S)

    zero_temp = False

    if T == 0.0:
        T = 1.0
        zero_temp = True

    # NOTE: the only purpose of this loop is to avoid OOM errors.
    for i in range(0, L, chunk_size):
        start = i
        end = min(i + chunk_size, L)
        query_chunk = query[:, start:end, :, :]

        attn_scores_chunk = einsum(query_chunk, key, "N L H E, N S H E -> N L H S")
        logits = scale_factor * attn_scores_chunk
        attn_weights_chunk = torch.softmax(logits / T, dim=-1)
        avg_attn_weights[:, start:end, :] = reduce(
            attn_weights_chunk,
            "N L H S -> N L S",
            reduction='mean',
            # reduction=torch.logsumexp,
        )

    if zero_temp:
        indices = torch.argmax(avg_attn_weights, dim=-1)
        avg_attn_weights = F.one_hot(indices, S).to(logits.dtype)

    return avg_attn_weights


def compute_hard_simil_masks(
    query,
    key,
    face_masks,
    grid_sizes,
    use_rope=False,
    freqs=None,
    chunk_size=512,
):
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

    attn_weights = compute_attn_weights(
        query,
        key_first_frame,
        # T=0.0,
        chunk_size=chunk_size,
    )

    background_mask = ~(face_masks.any(dim=0, keepdim=True))
    all_masks = torch.cat([background_mask, face_masks], dim=0)

    S = einsum(all_masks.float(), attn_weights, "P S, N L S -> N P L")

    inds = S.argmax(dim=1, keepdim=True)
    S = torch.zeros_like(S).scatter_(1, inds, 1.0)

    mask_zero = torch.isclose(S, torch.tensor(0.0, dtype=S.dtype, device=S.device))
    mask_one = torch.isclose(S, torch.tensor(1.0, dtype=S.dtype, device=S.device))
    assert (mask_zero | mask_one).all()

    sums = S.sum(dim=1)
    assert torch.allclose(sums, torch.tensor(1.0, dtype=sums.dtype, device=S.device)), (
        f"sums range: {sums.min()} {sums.max()}"
    )

    return S


# TODO: make temperature configurable?
def compute_soft_simil_masks(query, key, face_masks, grid_sizes, chunk_size=512):
    [[T, H, W]] = grid_sizes

    key_first_frame = rearrange(
        key, "1 (T H W) num_heads E -> 1 T (H W) num_heads E", T=T, H=H, W=W
    )[:, 0]

    attn_weights = compute_attn_weights(query, key_first_frame, chunk_size=chunk_size)

    background_mask = ~(face_masks.any(dim=0, keepdim=True))
    all_masks = torch.cat([background_mask, face_masks], dim=0)

    # apply the masks to the attention masks
    # P is for 'people' (it means # of masks)
    S = einsum(all_masks.float(), attn_weights, "P S, N L S -> N P L")
    S = torch.softmax(S, dim=1)

    sums = S.sum(dim=1)
    assert torch.allclose(sums, torch.tensor(1.0, dtype=sums.dtype, device=S.device)), (
        f"sums range: {sums.min()} {sums.max()}"
    )

    return S


