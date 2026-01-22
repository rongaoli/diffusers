try:
    import torch
except ImportError:
    torch = None

def is_torch_dist_rank_zero() -> bool:
    if torch is None:
        return True

    dist_module = getattr(torch, "distributed", None)
    if dist_module is None or not dist_module.is_available():
        return True

    if not dist_module.is_initialized():
        return True

    try:
        return dist_module.get_rank() == 0
    except (RuntimeError, ValueError):
        return True
