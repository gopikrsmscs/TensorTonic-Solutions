import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    mu = np.mean(x, axis = -1, keepdims = True)
    std = np.std(x, axis = -1, keepdims = True)
    num = x - mu 
    den = np.sqrt((std**2 + eps))
    y = gamma * (num/den) + beta
    return y