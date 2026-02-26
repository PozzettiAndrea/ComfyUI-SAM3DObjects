"""Lightweight VRAM logging helper. Prints current GPU memory usage."""
import torch

def vram(label: str) -> None:
    """Print current VRAM usage with a label."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() // (1024 * 1024)
    resv = torch.cuda.memory_reserved() // (1024 * 1024)
    free, total = torch.cuda.mem_get_info()
    free_mib = free // (1024 * 1024)
    total_mib = total // (1024 * 1024)
    used_mib = total_mib - free_mib
    print(f"[VRAM] {label}: torch={alloc}MiB alloc, {resv}MiB reserved | gpu={used_mib}MiB/{total_mib}MiB used")
