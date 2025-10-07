import math

import torch


def generate_soft_mask(
    mask: torch.Tensor, transition_duration: int, transition_type: str = "smooth"
) -> torch.Tensor:
    """
    Generates a 'soft' mask of floats from a 1D boolean PyTorch tensor.
    """

    if not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("The 'mask' must be a 1D boolean PyTorch tensor.")
    if not isinstance(transition_duration, int) or transition_duration < 1:
        return mask.to(torch.float32)

    device = mask.device
    soft_mask = torch.zeros_like(mask, dtype=torch.float32, device=device)
    mask_int = mask.to(torch.int32)

    padding = torch.tensor([0], dtype=torch.int32, device=device)
    padded_mask_int = torch.cat((padding, mask_int, padding))
    differences = torch.diff(padded_mask_int)
    start_indices = (differences == 1).nonzero().squeeze(dim=1)
    end_indices = (differences == -1).nonzero().squeeze(dim=1)

    for start, end in zip(start_indices, end_indices):
        soft_mask[start:end] = 1.0

    # --- Refined Ramp Generation (No slicing needed) ---
    D = transition_duration
    if transition_type == "linear":
        # Directly generate D points in the open interval (0, 1)
        ramp = torch.arange(1, D + 1, device=device) / (D + 1)
    elif transition_type == "smooth":
        # Directly generate D angles in the open interval (0, pi)
        x = (torch.arange(1, D + 1, device=device) * math.pi) / (D + 1)
        ramp = 0.5 * (1 - torch.cos(x))
    else:
        raise ValueError("transition_type must be either 'linear' or 'smooth'")

    reversed_ramp = torch.flip(ramp, dims=[0])

    # The rest of the logic is unchanged as it is correct
    for start in start_indices:
        ramp_start_index = max(0, start.item() - D)
        effective_ramp_length = start.item() - ramp_start_index
        if effective_ramp_length > 0:
            soft_mask[ramp_start_index:start] = torch.maximum(
                soft_mask[ramp_start_index:start], ramp[-effective_ramp_length:]
            )

    for end in end_indices:
        ramp_end_index = min(len(soft_mask), end.item() + D)
        effective_ramp_length = ramp_end_index - end.item()
        if effective_ramp_length > 0:
            soft_mask[end:ramp_end_index] = torch.maximum(
                soft_mask[end:ramp_end_index], reversed_ramp[:effective_ramp_length]
            )

    return soft_mask
