"""Operator implementations."""

import math
from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp


# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *



class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        dc_da=b*(a**(b-1))
        dc_db=(a**b)*log(a)
        return (out_grad*dc_da,out_grad*dc_db)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad*self.scalar*(node.inputs[0]**(self.scalar-1)))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        #return array_api.true_divide(a,b,dtype="float32")
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        return (out_grad/b,(out_grad*(-1)*a)/(b**2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        #return array_api.true_divide(a, self.scalar,dtype=a.dtype)
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        l=len(a.shape)
        temp=[i for i in range(l)]
        if self.axes is None:
          temp[l-1]=l-2
          temp[l-2]=l-1
          return a.permute(tuple(temp))
        temp[self.axes[0]]=self.axes[1]
        temp[self.axes[1]]=self.axes[0]
        return a.permute(tuple(temp))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (transpose(out_grad,self.axes))
        ### END YOUR SOLUTION



def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(),self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (reshape(out_grad,node.inputs[0].shape))
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        out_shape=out_grad.shape
        lo=len(out_shape)
        li=len(input_shape)
        delta=lo-li
        expanded_shape=list(range(lo-li))
        for i in range(li):
          if input_shape[i]==1 and out_shape[i+delta]>1:
            expanded_shape.append(i+delta)
        return reshape(summation(out_grad,tuple(expanded_shape)),input_shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
      #(3,2,4,5)->(2,5)maybe can't broadcast directly
        ### BEGIN YOUR SOLUTION
        oshape=node.inputs[0].shape
        shapel=len(oshape)
        before_shape=[0]*shapel
        judge=isinstance(self.axes,int)
        for i in range(shapel):
            if self.axes is None:
                before_shape=[1]
                break
            if (judge and i==self.axes) or ((not judge) and i in self.axes):
                before_shape[i]=1
            else:
                before_shape[i]=oshape[i]
        return broadcast_to(reshape(out_grad,tuple(before_shape)),oshape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    """Multiply (..., K) activations by a shared (K, N) weight matrix."""
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        a_shape = a.shape
        if len(a.shape) > 2:
            batch_size = 1
            for i in range(0, len(a.shape) - 1):
                batch_size *= a.shape[i]
            a = a.reshape((batch_size, a_shape[-1]))
        out = a@b
        if len(a_shape) > 2:
            out = out.reshape((*a_shape[:-1], b.shape[-1]))
        return out

    def gradient(self, out_grad, node):
        a, b = node.inputs
        if len(a.shape) == 2:
            return matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        # The forward flattens batch/sequence axes. Use the same convention
        # for the weight gradient, summing over all token positions at once.
        rows = math.prod(a.shape[:-1])
        a_flat = reshape(a, (rows, a.shape[-1]))
        grad_flat = reshape(out_grad, (rows, b.shape[1]))
        da = reshape(matmul(grad_flat, transpose(b)), a.shape)
        db = matmul(transpose(a_flat), grad_flat)
        return da, db


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1*a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad*-1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad/node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (exp(node.inputs[0])*out_grad)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        n=node.inputs[0]
        a=n.realize_cached_data()
        return out_grad*Tensor(a>0,device=n.device, dtype=n.dtype,requires_grad=False)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x=node.inputs[0]
        return out_grad*(tanh(x)**2-1)*(-1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        number=len(args)
        s=len(args[0].shape)+1
        new_shape=list(args[0].shape)
        new_shape.insert(self.axis,number)
        temp=array_api.empty(shape=tuple(new_shape),device=args[0].device)
        for i in range(number):
            q=[slice(None)]*s
            q[self.axis]=i
            temp[tuple(q)]=args[i]
        return temp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad,self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        l=A.shape[self.axis]
        s=len(A.shape)
        new_shape=list(A.shape)
        new_shape.pop(self.axis)
        temp=[]
        for i in range(l):
            q=[slice(None)]*s
            q[self.axis]=i
            temp.append((A[tuple(q)]).compact().reshape(new_shape))
        return tuple(temp)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shapes=list(a.shape)
        length=len(shapes)
        slices=[slice(None)]*length
        for i in self.axes:
            shapes[i]*=(1+self.dilation)
            slices[i]=slice(0,shapes[i],1+self.dilation)
        result=array_api.empty(shape=shapes,dtype=a.dtype,device=a.device)
        result.fill(0.0)
        result[tuple(slices)]=a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shapes=list(a.shape)
        length=len(shapes)
        slices=[slice(None)]*length
        for i in self.axes:
            slices[i]=slice(0,shapes[i],1+self.dilation)
            shapes[i]//=(1+self.dilation)
        result=array_api.empty(shape=shapes,dtype=a.dtype,device=a.device)
        result.fill(0.0)
        result=a[tuple(slices)]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        process_pad=[(0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)]
        A_padded=A.pad(tuple(process_pad))
        B=B.compact()
        N,H,W,C=A_padded.shape
        K,K1,_,out=B.shape
        Ns,Hs,Ws,Cs=A_padded.strides
        inner=K*K1*C
        reshaped=(N,(H-K)//self.stride+1,(W-K1)//self.stride+1,K,K1,C)
        produc=array_api.prod(reshaped)
        A_reshaped=A_padded.as_strided(shape=reshaped,strides=(Ns,Hs*self.stride,Ws*self.stride,Hs,Ws,Cs)).compact().reshape((produc//inner,inner))
        output=A_reshaped@B.reshape((inner,out))
        #print(output.shape)
        temp=(N,(H-K)//self.stride+1,(W-K1)//self.stride+1,out)
        #print(temp)
        return output.reshape(temp).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A,B=node.inputs
        K,K1,CIN,COUT=B.shape
        _,H,W,_=A.shape
        N,HP,WP,_=out_grad.shape
        #print(A.shape)
        #print(out_grad.shape)
        fliped=flip(B,(0,1))
        x_input=dilate(out_grad,axes=(1,2),dilation=self.stride-1)
        w_input=permute(A,(3,1,2,0))
        x_grad=conv(x_input,transpose(fliped),padding=K-1-self.padding)
        w_grad1=conv(w_input,permute(x_input,(1,2,0,3)),padding=self.padding)
        #print(w_grad1.shape)
        w_grad=permute(w_grad1,(1,2,0,3))
        return x_grad,w_grad
        ### END YOUR SOLUTION
def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
# rule to provide full dimension for simplicity
# understand of permute->inverse
class Permute(TensorOp):
    def __init__(self,axes:tuple[int]):
        self.axes=axes
    def compute(self, A):
        return A.permute(self.axes)
    def gradient(self,out_grad,node):
        inverse_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inverse_axes[ax] = i
        return permute(out_grad, tuple(inverse_axes))
def permute(A,axes):
    return Permute(axes)(A)


class FlashAttention(TensorOp):
    """Fused scaled dot-product attention (FlashAttention).

    Inputs: Q, K, V — all shape (batch, num_heads, seq_len, dim_head)
    Returns: O — shape (batch, num_heads, seq_len, dim_head)

    Forward:  O = softmax(Q @ K^T * scale [+ causal_mask]) @ V
    Backward: analytic gradient computed via NDArray-level numpy ops.
    """

    def __init__(self, causal: bool = False, softmax_scale: float = 1.0):
        self.causal = causal
        self.softmax_scale = softmax_scale

    def compute(self, q, k, v):
        return q.flash_attention(k, v,
                                 causal=self.causal,
                                 softmax_scale=self.softmax_scale)

    def gradient(self, out_grad, node):
        q, k, v = [inp.realize_cached_data() for inp in node.inputs]
        dO = out_grad.realize_cached_data()

        B, H, N, D = q.shape
        scale = self.softmax_scale

        # Convert to numpy for batched linear algebra
        q_np = q.numpy().reshape(B, H, N, D)
        k_np = k.numpy().reshape(B, H, N, D)
        v_np = v.numpy().reshape(B, H, N, D)
        dO_np = dO.numpy().reshape(B, H, N, D)

        # ── recompute forward: S, P ──
        S_np = q_np @ k_np.transpose(0, 1, 3, 2)  # (B, H, N, N)
        S_np *= scale
        if self.causal:
            mask = np.triu(np.ones((N, N), dtype=np.float32) * (-np.inf), 1)
            S_np += mask.reshape(1, 1, N, N)

        S_max = S_np.max(axis=-1, keepdims=True)
        S_exp = np.exp(S_np - S_max)
        P_np = S_exp / S_exp.sum(axis=-1, keepdims=True)

        # ── gradients ──
        # dP = dO @ V^T
        dP_np = dO_np @ v_np.transpose(0, 1, 3, 2)  # (B, H, N, N)

        # dS = P * (dP - sum(dP * P, axis=-1, keepdims=True))  (softmax grad)
        dS_np = P_np * (dP_np - (dP_np * P_np).sum(axis=-1, keepdims=True))

        # dQ = dS @ K * scale
        dQ_np = (dS_np @ k_np) * scale  # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)

        # dK = dS^T @ Q * scale
        dK_np = dS_np.transpose(0, 1, 3, 2) @ q_np * scale

        # dV = P^T @ dO
        dV_np = P_np.transpose(0, 1, 3, 2) @ dO_np

        from ..autograd import Tensor
        dQ = Tensor(dQ_np.reshape(B * H * N * D), device=q.device, dtype=q.dtype)
        dK = Tensor(dK_np.reshape(B * H * N * D), device=k.device, dtype=k.dtype)
        dV = Tensor(dV_np.reshape(B * H * N * D), device=v.device, dtype=v.dtype)

        return (dQ, dK, dV)


def flash_attention(q, k, v, causal=False, softmax_scale=1.0):
    return FlashAttention(causal=causal, softmax_scale=softmax_scale)(q, k, v)


class FusedLayerNorm(TensorOp):
    """Fused LayerNorm: out = weight * (x - mean) / sqrt(var + eps) + bias."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def compute(self, x, weight, bias):
        return x.layernorm(weight, bias, eps=self.eps)

    def gradient(self, out_grad, node):
        x, weight, bias = node.inputs
        N, D = x.shape

        x_np = x.realize_cached_data().numpy().reshape(N, D)
        w_np = weight.realize_cached_data().numpy().reshape(D)
        dO_np = out_grad.realize_cached_data().numpy().reshape(N, D)

        mean = x_np.mean(axis=1, keepdims=True)
        diff = x_np - mean
        var = (diff * diff).mean(axis=1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)

        x_hat = diff * inv_std

        # dX
        dx_hat = dO_np * w_np
        dvar = (dx_hat * diff).sum(axis=1, keepdims=True) * (-0.5) * (inv_std ** 3)
        dmean = dx_hat.sum(axis=1, keepdims=True) * (-inv_std) + dvar * (-2.0 / D) * diff.sum(axis=1, keepdims=True)
        dx_np = dx_hat * inv_std + dvar * (2.0 / D) * diff + dmean / D

        # dW
        dw_np = (dO_np * x_hat).sum(axis=0)

        # dB
        db_np = dO_np.sum(axis=0)

        from ..autograd import Tensor
        dX = Tensor(dx_np.reshape(N * D), device=x.device, dtype=x.dtype)
        dW = Tensor(dw_np.reshape(D), device=weight.device, dtype=weight.dtype)
        dB = Tensor(db_np.reshape(D), device=bias.device, dtype=bias.dtype)
        return (dX, dW, dB)


def layernorm(x, weight, bias, eps=1e-5):
    return FusedLayerNorm(eps=eps)(x, weight, bias)


class RNNFusedOp(TensorOp):
    """Fused multi-layer RNN: forward in one op at NDArray level.

    Inputs: X, W_ih_0, W_hh_0, b_ih_0, b_hh_0, ..., W_ih_L, W_hh_L, b_ih_L, b_hh_L, h0
    Returns: output (seq_len, bs, hidden_size) — last layer at each time step
    The gradient is computed via RNNFusedBackwardOp which does BPTT at NDArray level.
    """
    def __init__(self, num_layers, hidden_size, nonlinearity='tanh'):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

    def compute(self, *args):
        X = args[0]
        h0 = args[-1]
        nl, H = self.num_layers, self.hidden_size
        seq_len, bs = X.shape[0], X.shape[1]
        all_slice = slice(None)
        layer_input = X

        for l in range(nl):
            W_ih = args[1 + l * 4]
            W_hh = args[1 + l * 4 + 1]
            b_ih = args[1 + l * 4 + 2]
            b_hh = args[1 + l * 4 + 3]

            # h0[l] with proper slice: h0[l, :, :] -> (1, bs, H) -> (bs, H)
            h = h0[l, all_slice, all_slice].compact().reshape((bs, H))
            in_l = layer_input.shape[-1]

            # Batched input projection
            X_flat = layer_input.compact().reshape((seq_len * bs, in_l))
            ih_proj = (X_flat @ W_ih.compact()).compact().reshape((seq_len, bs, H))
            bias = (b_ih + b_hh).reshape((1, H)).broadcast_to((bs, H))

            out_l = NDArray.make((seq_len, bs, H), device=X.device)
            for t in range(seq_len):
                # ih_proj[t] -> (1, bs, H) -> (bs, H)
                ih_t = ih_proj[t, all_slice, all_slice].compact().reshape((bs, H))
                gate = ih_t + h @ W_hh.compact() + bias
                h = gate.tanh() if self.nonlinearity == 'tanh' else gate.maximum(0.0)
                out_l[t, all_slice, all_slice] = h
            layer_input = out_l

        return layer_input

    def gradient(self, out_grad, node):
        nl, H = self.num_layers, self.hidden_size
        result = RNNFusedBackwardOp(nl, H, self.nonlinearity)(out_grad, *node.inputs)
        # result is a TensorTuple; extract individual Tensors to return as Python tuple
        return tuple(result[i] for i in range(len(result)))


class RNNFusedBackwardOp(TensorTupleOp):
    """BPTT for fused RNN — all computation at NDArray level.

    Inputs: d_output, X, W_ih_0, W_hh_0, b_ih_0, b_hh_0, ..., h0
    Returns: gradient NDArrays for each input
    """
    def __init__(self, num_layers, hidden_size, nonlinearity='tanh'):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

    def compute(self, *args):
        d_output = args[0]        # (seq_len, bs, H)
        X = args[1]               # (seq_len, bs, in_size)
        nl, H = self.num_layers, self.hidden_size
        h0 = args[-1]
        seq_len, bs = X.shape[0], X.shape[1]
        device = X.device
        A = slice(None)

        # -- Recompute forward to get h[t] for all layers --
        layer_input = X
        all_out = [None] * nl          # each: (seq_len, bs, H)

        for l in range(nl):
            W_ih = args[2 + l * 4]
            W_hh = args[2 + l * 4 + 1]
            b_ih = args[2 + l * 4 + 2]
            b_hh = args[2 + l * 4 + 3]

            in_l = layer_input.shape[-1]
            X_flat = layer_input.compact().reshape((seq_len * bs, in_l))
            ih_proj = (X_flat @ W_ih.compact()).compact().reshape((seq_len, bs, H))
            bias = (b_ih + b_hh).reshape((1, H)).broadcast_to((bs, H))
            h = h0[l, A, A].compact().reshape((bs, H))
            out_l = NDArray.make((seq_len, bs, H), device=device)

            for t in range(seq_len):
                ih_t = ih_proj[t, A, A].compact().reshape((bs, H))
                gate = ih_t + h @ W_hh.compact() + bias
                h = gate.tanh() if self.nonlinearity == 'tanh' else gate.maximum(0.0)
                out_l[t, A, A] = h
            all_out[l] = out_l
            layer_input = out_l

        # -- BPTT: backward through layers (reverse order) --
        # Input layout: [X, W_ih0, W_hh0, b_ih0, b_hh0, W_ih1, ..., h0]
        # Total: 1 + 4*nl + 1
        num_inputs = 2 + 4 * nl
        grad_list = [None] * num_inputs
        d_prev_layer = d_output

        for l in range(nl - 1, -1, -1):
            W_ih = args[2 + l * 4]
            W_hh = args[2 + l * 4 + 1]
            b_ih = args[2 + l * 4 + 2]
            b_hh = args[2 + l * 4 + 3]
            in_l = X.shape[-1] if l == 0 else H
            layer_in = X if l == 0 else all_out[l - 1]
            h_t_vals = all_out[l]

            dW_ih_acc = self._zeros(W_ih.shape, device)
            dW_hh_acc = self._zeros(W_hh.shape, device)
            db_acc = self._zeros(b_ih.shape, device)
            dX_out = NDArray.make((seq_len, bs, in_l), device=device)
            dh = self._zeros((bs, H), device)

            for t in range(seq_len - 1, -1, -1):
                d_h = d_prev_layer[t, A, A].compact().reshape((bs, H))
                d_h = d_h + dh  # sum all contributions to h[t] first

                if self.nonlinearity == 'tanh':
                    h_t_c = h_t_vals[t, A, A].compact().reshape((bs, H))
                    d_gate = d_h * (1.0 + (-1.0) * h_t_c * h_t_c)
                else:
                    d_gate = d_h

                li_t = layer_in[t, A, A].compact().reshape((bs, in_l))
                dW_ih_acc = dW_ih_acc + li_t.permute((1, 0)) @ d_gate

                h_prev = h0[l, A, A] if t == 0 else h_t_vals[t - 1, A, A]
                if h_prev.ndim > 2:
                    h_prev = h_prev.compact().reshape((bs, H))
                else:
                    h_prev = h_prev.compact()
                dW_hh_acc = dW_hh_acc + h_prev.permute((1, 0)) @ d_gate

                db_acc = db_acc + d_gate.sum(axis=0)

                dX_out[t, A, A] = d_gate @ W_ih.compact().permute((1, 0))

                dh = d_gate @ W_hh.compact().permute((1, 0))

            # Store gradients at correct indices
            # Inputs: [0:X, 1:W_ih, 2:W_hh, 3:b_ih, 4:b_hh, ... , -1:h0]
            w_base = 1 + l * 4
            grad_list[w_base] = dW_ih_acc       # W_ih
            grad_list[w_base + 1] = dW_hh_acc   # W_hh
            grad_list[w_base + 2] = db_acc      # b_ih = db_acc
            grad_list[w_base + 3] = db_acc      # b_hh = db_acc (same)

            if l == 0:
                grad_list[0] = dX_out           # dX at index 0
            else:
                d_prev_layer = dX_out

        # dh0 at last index
        dh0 = NDArray.make((nl, bs, H), device=device)
        z = self._zeros((bs, H), device)
        for l in range(nl):
            dh0[l, A, A] = z
        grad_list[-1] = dh0

        return tuple(grad_list)

    @staticmethod
    def _zeros(shape, device):
        z = NDArray.make(shape, device=device)
        z.fill(0.0)
        return z
