import numpy as np


__device_name__ = "numpy"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


class Array:
    def __init__(self, size):
        self.array = np.empty(size, dtype=np.float32)

    @property
    def size(self):
        return self.array.size


def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
    )


def from_numpy(a, out):
    out.array[:] = a.flatten()


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    out.array[:] = a.array + b.array


def scalar_add(a, val, out):
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    out.array[:] = np.tanh(a.array)


def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)


def flash_attention(q, k, v, out, B, H, N, D, causal, softmax_scale):
    """
    FlashAttention reference implementation (numpy backend).

    Args:
        q, k, v: compact Array handles, each of shape (B, H, N, D)
        out: output Array handle, shape (B, H, N, D)
        B: batch size
        H: number of heads
        N: sequence length
        D: head dimension
        causal: whether to apply causal mask
        softmax_scale: 1/sqrt(D)
    """
    q_arr = q.array.reshape(B, H, N, D)
    k_arr = k.array.reshape(B, H, N, D)
    v_arr = v.array.reshape(B, H, N, D)

    # S = Q @ K^T * scale  ->  (B, H, N, N)
    S = q_arr @ k_arr.transpose(0, 1, 3, 2)
    S *= softmax_scale

    if causal:
        mask = np.triu(np.ones((N, N), dtype=np.float32) * (-np.inf), 1)
        S += mask.reshape(1, 1, N, N)

    # stable softmax along last dim
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)

    # O = P @ V  ->  (B, H, N, D)
    O = P @ v_arr

    out.array[:] = O.reshape(-1)


def layernorm(x, weight, bias, out, N, D, eps):
    """
    Fused LayerNorm reference (numpy backend).

    Args:
        x: compact array of shape (N, D)
        weight: compact array of shape (D,)
        bias: compact array of shape (D,)
        out: output array of shape (N, D)
        N: batch dimension (number of rows)
        D: feature dimension
        eps: epsilon
    """
    x_arr = x.array.reshape(N, D)
    w_arr = weight.array.reshape(D)
    b_arr = bias.array.reshape(D)

    mean = x_arr.mean(axis=1, keepdims=True)
    var = ((x_arr - mean) ** 2).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    out_arr = w_arr * (x_arr - mean) * inv_std + b_arr

    out.array[:] = out_arr.reshape(-1)
