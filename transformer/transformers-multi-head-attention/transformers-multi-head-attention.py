import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    each_dimension = W_q.shape[0]/num_heads
    (B,T,C) = Q.shape
    new_c = C//num_heads
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v
    Q = Q.reshape(Q.shape[0], Q.shape[1], num_heads, Q.shape[2]//num_heads).transpose(0,2,1,3)
    K = K.reshape(K.shape[0], K.shape[1], num_heads, K.shape[2]//num_heads).transpose(0,2,1,3)
    V = V.reshape(V.shape[0], V.shape[1], num_heads, V.shape[2]//num_heads).transpose(0,2,1,3)

    scores = Q @ K.transpose(0,1,3,2)
    scores = scores/np.sqrt(new_c)
    scores = scores - scores.max(axis = -1, keepdims = True)
    attention = np.exp(scores)
    attention = attention/attention.sum(axis =-1, keepdims = True)
    output = attention @ V 
    out = output.transpose(0,2,1,3).reshape(B,T,C)
    return out @ W_o
    
    