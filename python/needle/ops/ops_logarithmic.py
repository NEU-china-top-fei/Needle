from typing import Optional, Any, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *
from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class LogSoftmax(TensorOp):
    def __init__(self):
        self.axes = (1,)

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxi = Z.max(axis=self.axes, keepdims=True)
        maxi_bc = array_api.broadcast_to(maxi, Z.shape)
        exp_val = array_api.exp(Z - maxi_bc)
        sum_val = array_api.sum(exp_val, axis=self.axes, keepdims=True)
        lse = array_api.log(sum_val) + maxi
        lse_bc = array_api.broadcast_to(lse, Z.shape)
        return Z - lse_bc
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # node 是 LogSoftmax 的输出，即 log(softmax(Z))
        # 所以 exp(node) 就是 softmax(Z)
        softmax_z = exp(node)
        
        # 1. 计算 out_grad 在对应维度上的和
        # 如果 out_grad 是 (batch, dim)，sum 之后是 (batch,)
        grad_sum = out_grad.sum(axes=self.axes)
        
        # 2. 为了广播，需要把 grad_sum reshape 回 (batch, 1)
        # 构造新形状：将被缩减的轴设为 1
        new_shape = list(out_grad.shape)
        for axis in self.axes:
            new_shape[axis] = 1
        
        # 3. 最终公式：out_grad - softmax(Z) * sum(out_grad)
        return out_grad - softmax_z * grad_sum.reshape(tuple(new_shape))
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxi = Z.max(axis=self.axes, keepdims=True)
        if self.axes is None:
            maxi_bc = array_api.broadcast_to(array_api.reshape(maxi, (1,) * Z.ndim), Z.shape)
        else:
            maxi_bc = array_api.broadcast_to(maxi, Z.shape)
        exp_val = array_api.exp(Z - maxi_bc)
        sum_val = array_api.sum(exp_val, axis=self.axes)
        log_val = array_api.log(sum_val)
        if self.axes is None:
            log_val_r = array_api.reshape(log_val, (1,))
            maxi_r = array_api.reshape(maxi, (1,))
            return log_val_r + maxi_r
        else:
            axes = (self.axes,) if isinstance(self.axes, int) else self.axes
            new_shape = tuple(s for i, s in enumerate(maxi.shape) if i not in axes)
            return log_val + array_api.reshape(maxi, new_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        inputT = node.inputs[0]
        l = len(inputT.shape)
        s = [1] * l
        if self.axes is not None:
            axes = (self.axes,) if isinstance(self.axes, int) else self.axes
            for i in range(l):
                if i not in axes:
                    s[i] = inputT.shape[i]
        return out_grad.reshape(s).broadcast_to(inputT.shape) * (exp(inputT - node.reshape(tuple(s)).broadcast_to(inputT.shape)))
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)