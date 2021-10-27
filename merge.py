import torch

def merge(
    embeddings1: torch.Tensor, 
    embeddings2: torch.Tensor, 
    embeddings3: torch.Tensor) -> torch.Tensor:
    """
    Merge three embeddings.
    Three embeddings are with the same shape: [108, 49, 1024].
    """
    assert embeddings1.shape == embeddings2.shape and embeddings1.shape == embeddings3.shape,\
        'Three embeddings are not the same shape'
    # Interleave three embeddings.
    merge_embeddings = torch.stack((embeddings1, embeddings2, embeddings3), dim=3)
    # Reshape: [108, 49, 1024, 3] -> [108, 49, 3072], then return the Tensor.
    return torch.flatten(merge_embeddings, start_dim=2)