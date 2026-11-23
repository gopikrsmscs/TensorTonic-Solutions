import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d = Q.shape[2]
    K_T = K.transpose(-2,-1)
    upper = Q @ K_T
    lower = d **(0.5)
    attention = torch.softmax(upper/lower, dim = -1)
    return torch.matmul(attention, V)
    #pass