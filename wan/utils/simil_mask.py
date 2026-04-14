import math

import einops
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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


def check_mece(S) -> bool:
    mask_zero = torch.isclose(S, torch.tensor(0.0, dtype=S.dtype, device=S.device))
    mask_one = torch.isclose(S, torch.tensor(1.0, dtype=S.dtype, device=S.device))
    assert (mask_zero | mask_one).all()

    sums = S.sum(dim=1)

    return torch.allclose(sums, torch.tensor(1.0, dtype=sums.dtype, device=S.device))


def discretize_masks(soft_masks, threshold, bg_idx=0):

    max_scores, max_inds = torch.max(soft_masks, dim=1)
    is_valid_entity = (max_scores > threshold) & (max_inds != bg_idx)
    final_classes = torch.where(is_valid_entity, max_inds, bg_idx)

    hard_masks = torch.zeros_like(soft_masks)
    hard_masks.scatter_(dim=1, index=final_classes.unsqueeze(1), value=1.0)

    return hard_masks


def apply_gaussian_smooth(soft_masks, grid_sizes, kernel_size=5, sigma=1.0):
    [[T, H, W]] = grid_sizes

    # masks_2d = soft_masks.view(B, C, H, W)
    masks_2d = rearrange(soft_masks, "B C (T H W) -> B C T H W", T=T, H=H, W=W)

    smoothed_masks_2d = TF.gaussian_blur(
        masks_2d, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
    )

    # smoothed_masks = smoothed_masks_2d.view(B, C, L)
    smoothed_masks = rearrange(smoothed_masks_2d, "B C T H W -> B C (T H W)")

    return smoothed_masks


def compute_hard_simil_masks(
    query,
    key,
    face_masks,
    grid_sizes,
    use_rope=False,
    freqs=None,
    chunk_size=512,
    threshold=0.0,
    smooth=False,
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

    if smooth:
        S = apply_gaussian_smooth(S, grid_sizes)

    S_hat = discretize_masks(S, threshold)

    assert check_mece(S_hat)

    return S_hat


def compute_soft_simil_masks(
    query, key, face_masks, grid_sizes, temperature=1.0, chunk_size=512
):
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
    S = torch.softmax(S / temperature, dim=1)

    sums = S.sum(dim=1)
    assert torch.allclose(sums, torch.tensor(1.0, dtype=sums.dtype, device=S.device)), (
        f"sums range: {sums.min()} {sums.max()}"
    )

    return S


# def compute_soft_simil_masks_iterative(
#     query, key, face_masks, grid_sizes, chunk_size=256
# ):
#     [[T, H, W]] = grid_sizes
#     N_batch = query.shape[0]
#
#     query_frames = rearrange(
#         query, "N (T H W) num_heads E -> N T (H W) num_heads E", T=T, H=H, W=W
#     )
#     key_frames = rearrange(
#         key, "N (T H W) num_heads E -> N T (H W) num_heads E", T=T, H=H, W=W
#     )
#
#     q_batched = rearrange(
#         query_frames[:, 1:], "N T L num_heads E -> (N T) L num_heads E"
#     )
#     k_batched = rearrange(
#         key_frames[:, :-1], "N T L num_heads E -> (N T) L num_heads E"
#     )
#
#     # Output shape: ((N * T-1), L, S) where L = S = H*W. (Heads are averaged inside here)
#     attn_weights = compute_attn_weights(q_batched, k_batched, chunk_size=chunk_size)
#
#     attn_weights = rearrange(attn_weights, "(N T) L S -> N T L S", N=N_batch)
#
#     background_mask = ~(face_masks.any(dim=0, keepdim=True))
#     all_masks = torch.cat([background_mask, face_masks], dim=0).float()
#
#     S_0 = all_masks.unsqueeze(0).expand(N_batch, -1, -1)
#     S_0 = S_0 / (S_0.sum(dim=0, keepdim=True) + 1e-8)
#
#     S_frames = [S_0]
#     current_masks = S_0
#
#     for t in range(T - 1):
#         attn_t = attn_weights[:, t]  # Shape: (N, L, S)
#
#         S_t = einsum(current_masks, attn_t, "N P S, N L S -> N P L")
#         S_t = S_t / (S_t.sum(dim=1, keepdim=True) + 1e-8)
#
#         S_frames.append(S_t)
#
#         hard_mask_idx = S_t.argmax(dim=1, keepdim=True)
#         current_masks = torch.zeros_like(S_t).scatter_(1, hard_mask_idx, 1.0)
#
#     S = torch.cat(S_frames, dim=2)  # Shape: (N, P, T*H*W)
#
#     sums = S.sum(dim=1)
#     assert torch.allclose(sums, torch.tensor(1.0, dtype=sums.dtype, device=S.device)), (
#         f"sums range: {sums.min()} {sums.max()}"
#     )
#
#     # Remove background mask
#     soft_masks = S[:, 1:, :]
#
#     return soft_masks


@torch.no_grad()
def compute_soft_simil_masks_iterative(
    query, key, face_masks, grid_sizes, chunk_size=128
):
    [[T, H, W]] = grid_sizes
    N_batch = query.shape[0]

    # 1. Separate the time dimension
    query_frames = rearrange(
        query,
        "N (T H W) num_heads E -> N T (H W) num_heads E",
        N=N_batch,
        T=T,
        H=H,
        W=W,
    )
    key_frames = rearrange(
        key, "N (T H W) num_heads E -> N T (H W) num_heads E", N=N_batch, T=T, H=H, W=W
    )

    # 2. PREPARE KEYS FOR ANCHORING
    # Extract Frame 0 keys and expand them to match the T-1 temporal steps
    k_0 = key_frames[:, 0:1].expand(
        -1, T - 1, -1, -1, -1
    )  # Shape: (N, T-1, L, num_heads, E)
    k_prev = key_frames[:, :-1]  # Shape: (N, T-1, L, num_heads, E)

    # Concatenate Frame 0 and Frame t-1 keys along the spatial dimension L
    # The new spatial dimension is now 2*L
    k_combined = torch.cat([k_0, k_prev], dim=2)

    # 3. Batched Native Attention (Query attends to BOTH 0 and t-1)
    q_batched = rearrange(
        query_frames[:, 1:], "N T_minus_1 L num_heads E -> (N T_minus_1) L num_heads E"
    )
    k_batched = rearrange(
        k_combined, "N T_minus_1 Two_L num_heads E -> (N T_minus_1) Two_L num_heads E"
    )

    # The resulting attention weights will have shape (N*(T-1), L, 2*L)
    batched_attn_weights = compute_attn_weights(
        q_batched, k_batched, chunk_size=chunk_size
    )
    batched_attn_weights = rearrange(
        batched_attn_weights, "(N T_minus_1) L Two_L -> N T_minus_1 L Two_L", N=N_batch
    )

    # 4. Prepare initial perfect masks for Frame 0
    background_mask = ~(face_masks.any(dim=0, keepdim=True))
    all_masks = torch.cat([background_mask, face_masks], dim=0).float()

    S_0 = all_masks.unsqueeze(0).expand(N_batch, -1, -1)
    S_0 = S_0 / (S_0.sum(dim=1, keepdim=True) + 1e-8)

    S_frames = [S_0]
    current_masks = S_0  # S_{t-1}

    # ---------------------------------------------------------
    # ANCHORED SEQUENTIAL MASK PROPAGATION
    # ---------------------------------------------------------
    for t in range(T - 1):
        attn_t = batched_attn_weights[:, t]  # Shape: (N, L, 2*L)

        # Concatenate the perfect Frame 0 mask with the current Frame t-1 mask
        # combined_masks shape: (N, P, 2*L)
        combined_masks = torch.cat([S_0, current_masks], dim=2)

        # Propagate to get the new SOFT mask
        # Attention maps the 2*L spatial keys back into the L spatial queries
        S_t_soft = einsum(combined_masks, attn_t, "N P Two_L, N L Two_L -> N P L")

        # Normalize so probabilities sum to 1
        S_t_soft = S_t_soft / (S_t_soft.sum(dim=1, keepdim=True) + 1e-8)

        S_frames.append(S_t_soft)

        # Update current masks for the next step
        current_masks = S_t_soft

    # Concatenate sequence along the spatial/temporal dimension L
    S = torch.cat(S_frames, dim=2)  # Shape: (N, P, T*H*W)

    return S
