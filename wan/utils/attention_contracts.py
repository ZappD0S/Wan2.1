"""Small, executable contracts shared by the custom attention paths."""

import torch
from torch.nn import functional as F

MASK_SHARING_ALIASES = {
    None: "current",
    "none": "current",
    "current": "current",
    "first": "first",
    "block0": "first",
    "prev": "previous_generated",
    "previous-generated": "previous_generated",
}


def normalize_mask_sharing(value):
    """Return the canonical generated-mask sharing policy."""
    try:
        return MASK_SHARING_ALIASES[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "mask_sharing must be one of none/current, first/block0, or "
            "prev/previous-generated"
        ) from exc


def select_used_simil_masks(generated, incoming):
    """Select the one mask tensor that both self- and cross-attention must use."""
    return generated if incoming is None else incoming


def update_shared_simil_masks(policy, block_index, generated, active):
    """Return the incoming shared mask for the next block."""
    if policy == "previous_generated" or (policy == "first" and block_index == 0):
        return generated
    return active


def self_attention_bias_enabled(bias_kwargs):
    """Apply self-routing only when every schedule/CFG guard enables it."""
    return (
        bool(bias_kwargs["bias"])
        and bias_kwargs["bias_method"] != "none"
        and bool(bias_kwargs["self_attention_masking"])
    )


def build_static_simil_masks(face_masks, T):
    """Build a background+subject partition without invoking a tracker."""
    if face_masks.ndim != 2:
        raise ValueError("flattened face_masks must have shape [subjects, H*W]")
    if (face_masks.bool().sum(dim=0) > 1).any():
        raise ValueError("subject masks must be disjoint")
    face_masks = face_masks.bool()
    background_mask = ~face_masks.any(dim=0, keepdim=True)
    all_masks = torch.cat([background_mask, face_masks], dim=0)
    return all_masks.unsqueeze(0).repeat(1, 1, T).float()


def validate_simil_masks(masks, expected_classes=None):
    """Require floating, finite class partitions for fixed/hard/soft masks."""
    if masks.ndim != 3:
        raise ValueError("similarity masks must have shape [batch, classes, tokens]")
    if expected_classes is not None and masks.shape[1] != expected_classes:
        raise ValueError("similarity mask class count does not match subjects+background")
    masks = masks.float()
    if not torch.isfinite(masks).all() or (masks < 0).any() or (masks > 1).any():
        raise ValueError("similarity masks must be finite and lie in [0, 1]")
    sums = masks.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-2):
        raise ValueError("similarity mask classes must form a partition")
    return masks


def validate_eligibility(eligibility):
    """Validate continuous regional eligibility weights."""
    if not eligibility.is_floating_point():
        eligibility = eligibility.float()
    if not torch.isfinite(eligibility).all():
        raise ValueError("attention eligibility must be finite")
    if (eligibility < 0).any() or (eligibility > 1).any():
        raise ValueError("attention eligibility must lie in [0, 1]")
    return eligibility


def ediff_attention(
    q, k, v, eligibility, strength, normalized_timestep, valid_tokens=None
):
    """Finite eDiff-I-style positive bias; eligibility is not hard exclusion."""
    eligibility = validate_eligibility(eligibility).to(device=q.device, dtype=q.dtype)
    attn_bias = strength * normalized_timestep * eligibility
    if valid_tokens is not None:
        valid_tokens = valid_tokens.to(device=q.device, dtype=torch.bool)
        if valid_tokens.ndim != 1 or valid_tokens.numel() != k.shape[-2]:
            raise ValueError("valid_tokens must match the text key length")
        if not valid_tokens.any():
            raise ValueError("at least one text token must be valid")
        attn_bias = attn_bias.masked_fill(
            ~valid_tokens[None, None, :], float("-inf")
        )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(1))


def regional_attention(q, k, v, eligibility):
    """SDPA with log eligibility; exact zero excludes a key for that query.

    Fractional values are a continuous extension of binary regional prompting.
    Rows with no eligible keys intentionally produce a zero text-attention output.
    """
    eligibility = validate_eligibility(eligibility).to(device=q.device, dtype=q.dtype)
    has_eligible_key = eligibility.gt(0).any(dim=-1)
    safe_eligibility = eligibility.clone()
    if not has_eligible_key.all():
        # Avoid backend-dependent NaNs for an all-masked row, then zero that row.
        safe_eligibility[..., 0] = torch.where(
            has_eligible_key,
            safe_eligibility[..., 0],
            torch.ones_like(safe_eligibility[..., 0]),
        )
    attn_bias = torch.where(
        safe_eligibility > 0,
        safe_eligibility.log(),
        torch.full_like(safe_eligibility, float("-inf")),
    ).unsqueeze(1)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    return output * has_eligible_key[:, None, :, None].to(output.dtype)
