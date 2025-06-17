import numpy as np
import pytest

def quantize_int8(x):
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / 255 if x_max != x_min else 1.0
    q = np.round((x - x_min) / scale).astype(np.uint8)
    return q, x_min, scale

def dequantize_int8(q, x_min, scale):
    return q.astype(np.float32) * scale + x_min

def quantize_int4(x):
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / 15 if x_max != x_min else 1.0
    q = np.round((x - x_min) / scale).astype(np.uint8)
    # Pack two 4-bit values into one uint8
    if len(q.shape) == 1:
        if q.shape[0] % 2 != 0:
            q = np.append(q, 0)
        q4 = (q[::2] << 4) | (q[1::2])
        return q4, x_min, scale, q.shape[0]
    else:
        # For 2D arrays, flatten and reshape after
        orig_shape = q.shape
        flat = q.flatten()
        if flat.shape[0] % 2 != 0:
            flat = np.append(flat, 0)
        q4 = (flat[::2] << 4) | (flat[1::2])
        return q4, x_min, scale, orig_shape

def dequantize_int4(q4, x_min, scale, orig_shape):
    # Unpack two 4-bit values from each uint8
    flat_len = orig_shape if isinstance(orig_shape, int) else np.prod(orig_shape)
    q = np.zeros(flat_len, dtype=np.uint8)
    q[::2] = (q4 >> 4) & 0x0F
    q[1::2] = q4 & 0x0F
    x = q.astype(np.float32) * scale + x_min
    if isinstance(orig_shape, int):
        return x[:orig_shape]
    else:
        return x[:flat_len].reshape(orig_shape)

def cosine_similarity(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return np.mean(dot / (norm_a * norm_b + 1e-8))

def test_int4_vs_int8_quantization():
    np.random.seed(42)
    num_vectors = 100
    dim = 64
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)

    # int8 quantization
    q8, min8, scale8 = quantize_int8(vectors)
    deq8 = dequantize_int8(q8, min8, scale8)
    mse8 = np.mean((vectors - deq8) ** 2)
    cos8 = cosine_similarity(vectors, deq8)

    # int4 quantization
    q4, min4, scale4, orig_shape = quantize_int4(vectors)
    deq4 = dequantize_int4(q4, min4, scale4, orig_shape)
    mse4 = np.mean((vectors - deq4) ** 2)
    cos4 = cosine_similarity(vectors, deq4)

    print(f"int8 quantization:   MSE={mse8:.6f}, Cosine similarity={cos8:.6f}")
    print(f"int4 quantization:   MSE={mse4:.6f}, Cosine similarity={cos4:.6f}")

    # Assert int8 is more accurate than int4
    assert mse8 < mse4
    assert cos8 > cos4
    # Both should be reasonably close
    assert cos8 > 0.95
    assert cos4 > 0.85 