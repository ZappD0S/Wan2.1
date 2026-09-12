"""Diffusion-seed resolution with a distributed-safe public API contract."""

import random
import sys

import torch
import torch.distributed as dist


def resolve_diffusion_seed(seed: int, device) -> int:
    """Resolve one integer seed on rank zero and broadcast it before noise creation."""
    if not isinstance(seed, int):
        raise TypeError("diffusion seed must be an integer")
    if seed >= 2**63:
        raise ValueError("diffusion seed must fit in a signed 64-bit integer")
    if dist.is_initialized():
        rank = dist.get_rank()
        resolved = seed if seed >= 0 else (random.randint(0, sys.maxsize) if rank == 0 else 0)
        seed_tensor = torch.tensor([resolved], dtype=torch.long, device=device)
        dist.broadcast(seed_tensor, src=0)
        return int(seed_tensor.item())
    return seed if seed >= 0 else random.randint(0, sys.maxsize)


def initialize_diffusion_noise(seed: int, device, noise_factory):
    """Resolve/broadcast, then create noise while retaining generator state."""
    resolved = resolve_diffusion_seed(seed, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(resolved)
    noise = noise_factory(generator)
    return resolved, generator, noise
