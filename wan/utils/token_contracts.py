"""Token-mask helpers for concatenated independently encoded sentences."""

import torch


def prefix_for_concatenated_context(local_mask, preceding_context_lengths):
    """Place a sentence-local mask at its global concatenated-token offset."""
    if local_mask.ndim != 1:
        raise ValueError("token mask must be one-dimensional")
    offset = sum(int(length) for length in preceding_context_lengths)
    if offset < 0:
        raise ValueError("context lengths must be nonnegative")
    return torch.cat([local_mask.new_zeros(offset), local_mask])
