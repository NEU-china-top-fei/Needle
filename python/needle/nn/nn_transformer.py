from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
        use_flash_attn = False,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.use_flash_attn = use_flash_attn
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        if (self.use_flash_attn and hasattr(q.device, "flash_attention")
                and not (self.training and self.dropout.p > 0)):
            softmax_scale = 1.0 / np.sqrt(q_dim)
            result = ops.flash_attention(
                q, k, v,
                causal=self.causal,
                softmax_scale=softmax_scale,
            )
            probs = None
        else:
            pre_score=self.matmul(q,k)/np.sqrt(q_dim)
            if self.causal:
                pre_score+=self.create_causal_mask(queries_len,queries_len,device=self.device).broadcast_to((batch_size,num_head,queries_len,queries_len))
            probs=self.dropout(self.softmax(pre_score))
            result=self.matmul(probs,v.transpose())
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        use_flash_attn = False,
        use_layernorm = True,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype, use_layernorm=use_layernorm)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype, use_layernorm=use_layernorm)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype, use_layernorm=use_layernorm)

        inner_dim = num_head * dim_head

        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype,
            use_flash_attn=use_flash_attn)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        qprime=ops.permute(self.q_projection(self.prenorm_q(q.reshape((batch_size*queries_len,q_dim))).reshape((batch_size,queries_len,q_dim))).reshape((batch_size,queries_len,self.num_head,self.dim_head)),(0,2,1,3))
        kprime=ops.permute(self.k_projection(self.prenorm_k(k.reshape((batch_size*queries_len,q_dim))).reshape((batch_size,queries_len,q_dim))).reshape((batch_size,queries_len,self.num_head,self.dim_head)),(0,2,1,3))
        vprime=ops.permute(self.v_projection(self.prenorm_v(v.reshape((batch_size*queries_len,q_dim))).reshape((batch_size,queries_len,q_dim))).reshape((batch_size,queries_len,self.num_head,self.dim_head)),(0,2,1,3))
        x=ops.permute(self.attn(qprime,kprime,vprime)[0],(0,2,1,3)).reshape((batch_size,queries_len,self.num_head*self.dim_head))
        result=self.out_projection(x).reshape((batch_size,queries_len,self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        use_flash_attn = False,
        use_layernorm = True,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.multiattn=AttentionLayer(q_features,num_head,dim_head,causal=causal,device=device,dtype=dtype,dropout=dropout,use_flash_attn=use_flash_attn,use_layernorm=use_layernorm)
        self.drop=Dropout(dropout)
        self.norm=LayerNorm1d(q_features,device=device,dtype=dtype,use_layernorm=use_layernorm)
        self.linear1=Linear(q_features,hidden_size,device=device,dtype=dtype)
        self.linear2=Linear(hidden_size,q_features,device=device,dtype=dtype)
        self.nonlinear=ReLU()
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        temp=x+self.drop(self.multiattn(x))
        x=temp+self.drop(self.linear2(self.drop(self.nonlinear(self.linear1(self.norm(temp.reshape((batch_size*seq_len,x_dim))).reshape((batch_size,seq_len,x_dim)))))))
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048,
        use_flash_attn = False,
        use_layernorm = True,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.positionembed=Embedding(sequence_len,embedding_size,device=device,dtype=dtype)

        transformerlayer=[TransformerLayer(embedding_size,num_head=num_head,dim_head=dim_head,hidden_size=hidden_size,dropout=dropout,
                                              causal=causal,device=device,dtype=dtype,use_flash_attn=use_flash_attn,use_layernorm=use_layernorm) for _ in range(num_layers)]
        self.num_layers=num_layers
        self.sequence_len=sequence_len
        self.model=Sequential(*transformerlayer)
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        bs,seq_len,dim=x.shape
        time = np.repeat(np.arange(seq_len), bs).reshape((seq_len, bs)).T
        timestep=Tensor(time,device=self.device,dtype=self.dtype)
        timeembed=self.positionembed(timestep)
        into=x+timeembed
        x=self.model(into)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
