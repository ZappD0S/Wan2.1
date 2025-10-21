import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce


def compute_simil_masks(query, key, face_masks, chunk_size=512):
    def _weighted_average(x, weights, dim):
        weights = weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-6)
        return (x * weights).sum(dim=dim)

    batch_size, seq_len_q, num_heads, head_dim = query.shape
    seq_len_k = key.shape[1]

    sum_attn_weights = query.new_zeros(batch_size, seq_len_q, seq_len_k)

    scale_factor = 1 / math.sqrt(head_dim)

    for i in range(0, seq_len_q, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, seq_len_q)
        query_chunk = query[:, start_idx:end_idx, :, :]

        attn_scores_chunk = einsum(query_chunk, key, "N L H E, N S H E -> N L H S")
        attn_weights_chunk = F.softmax(attn_scores_chunk * scale_factor, dim=-1)

        summed_weights_chunk = reduce(attn_weights_chunk, "N L H S -> N L S", "sum")
        sum_attn_weights[:, start_idx:end_idx, :] += summed_weights_chunk

    attn_weights = sum_attn_weights / num_heads

    background_mask = ~(face_masks.any(dim=0, keepdim=True))
    face_masks = torch.cat([face_masks, background_mask])

    # we only want to take into account the effect of the tokens of the face in the first frame
    simil_scores = []
    for target_mask in face_masks:
        target_mask = rearrange(target_mask, "S -> 1 1 S")
        simil_scores.append(
            _weighted_average(attn_weights, weights=target_mask.float(), dim=2)
        )

    simil_scores = torch.stack(simil_scores, dim=1)
    simil_masks = simil_scores == simil_scores.max(dim=1, keepdim=True).values

    # we don't care about the background
    simil_masks = simil_masks[:, :-1]

    return simil_masks
