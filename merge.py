import torch

def merge(*embeddings) -> torch.Tensor:
    """
    Merge three embeddings.
    Three embeddings are with the same shape: [108, 49, 1024].
    """
    # Interleave three embeddings.
    merge_embeddings = torch.stack(embeddings, dim=3)
    # Reshape: [108, 49, 1024, 3] -> [108, 49, 3072], then return the Tensor.
    return torch.flatten(merge_embeddings, start_dim=2)