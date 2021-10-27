import torch

def merge(*embeddings) -> torch.Tensor:
    """
    Merge N 3-Dimensional embeddings.
    All embeddings should be the same shape.
    """
    # Interleave embeddings.
    merge_embeddings = torch.stack(embeddings, dim=embeddings[0].dim())
    # Reshape: [108, 49, 1024, 3] -> [108, 49, 3072], then return the Tensor.
    return torch.flatten(merge_embeddings, start_dim=(embeddings[0].dim()-1))

a = torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
# shape of a: [3, 3, 3], with all element=1
b = 2 * a
c = 3 * a
d = merge(a, b, c)
print(d)
print(f'Shape: a={a.shape}, b={b.shape}, c={c.shape}, d={d.shape}')