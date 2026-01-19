def mat_mul(a, b):
    assert len(a[0]) == len(b) # (m x n) x (n x p) = (m x p)
    M, N, P = len(a), len(b), len(b[0])
    result = []
    for m in range(M):
        row = []
        for p in range(P):
            pairs = zip(a[m], [b[n][p] for n in range(N)])
            mult = [x[0] * x[1] for x in pairs]
            val = sum(mult)
            row.append(val)
        result.append(row)
    return result


import torch
def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a)
        b = torch.tensor(b)
    if a.shape[1] != a.shape[0]:
        return torch.tensor(-1)
    return a @ b

