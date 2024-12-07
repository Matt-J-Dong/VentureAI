import os
import torch.distributed as dist

def is_torchrun_running():
    """
    Detect if torchrun is running on a group of GPUs.

    Returns:
        bool: True if torchrun is running, False otherwise.
    """
    # Check if torchrun-related environment variables are set
    torchrun_env_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE']
    if not all(var in os.environ for var in torchrun_env_vars):
        return False

    # Check if the PyTorch distributed process group is initialized
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

# Example usage
if __name__ == "__main__":
    if is_torchrun_running():
        print("torchrun is currently running.")
        print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")
        print(f"RANK: {os.environ['RANK']}")
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    else:
        print("torchrun is not running.")
