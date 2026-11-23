import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    # feed = x @ W1 + b1 
    # relu = np.maximum(0, feed)
    # ffn = relu @ W2 + b2 
    hidden = np.dot(x, W1) + b1 
    hidden = np.maximum(0,hidden)
    ffn = np.dot(hidden, W2) + b2 
    return ffn