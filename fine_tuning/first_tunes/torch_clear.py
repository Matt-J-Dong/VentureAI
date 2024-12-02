import torch

# Clear CUDA cache
torch.cuda.empty_cache()

# Delete unused variables (if applicable)
del model
del tokenized_datasets
del trainer
