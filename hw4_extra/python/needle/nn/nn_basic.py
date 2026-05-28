"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.kaiming_uniform(in_features,out_features,nonlinearity="relu",dtype=dtype,device=device))
        if bias:
          self.bias=Parameter(init.kaiming_uniform(out_features,1,"relu",dtype=dtype,device=device).reshape((1,self.out_features)))
        else:
          self.bias=None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        pro=ops.matmul(X,self.weight)
        if self.bias is not None:
          pro=pro+self.bias.broadcast_to(pro.shape)
        return pro
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        dim=1
        l=len(X.shape)
        for i in range(1,l,1):
          dim*=X.shape[i]
        return X.reshape((X.shape[0],dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for operation in self.modules:
          x=operation(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        encoded=init.one_hot(logits.shape[1],y,device=logits.device)
        return ops.summation(ops.logsumexp(logits,(1,))-ops.summation(logits*encoded,(1,)))/logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(dim,device=device,dtype=dtype))
        self.bias=Parameter(init.zeros(dim,device=device,dtype=dtype))
        self.running_mean=init.zeros(dim,device=device,dtype=dtype,requires_grad=False)
        self.running_var=init.ones(dim,device=device,dtype=dtype,requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size,num_feature=x.shape
        weight=self.weight.reshape((1,num_feature)).broadcast_to((batch_size,num_feature))
        bias=self.bias.reshape((1,num_feature)).broadcast_to((batch_size,num_feature))
        if self.training:
          e=x.sum(axes=(0,))/batch_size
          e_bc=e.reshape((1,num_feature)).broadcast_to((batch_size,num_feature))
          delta=x-e_bc
          var=(delta*delta).sum(axes=(0,))/(batch_size)
          var_bc=var.reshape((1,num_feature)).broadcast_to((batch_size,num_feature))
          de=(var_bc+self.eps)**0.5
          self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*e.data
          self.running_var=(1-self.momentum)*self.running_var+self.momentum*var.data
          return weight*delta/de+bias
        else:
          mu=self.running_mean.reshape((1,num_feature)).broadcast_to((batch_size,num_feature))
          sigma=self.running_var.reshape((1,num_feature)).broadcast_to((batch_size,num_feature))
          return weight*(x-mu)/((sigma+self.eps)**0.5)+bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(dim,1,device=device,dtype=dtype))
        self.bias=Parameter(init.zeros(dim,1,device=device,dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        bs,fn=x.shape
        expectation=(x.sum(axes=(1,))/fn).reshape((bs,1)).broadcast_to(x.shape)
        delta=x-expectation
        variance=((delta*delta).sum(axes=(1,))/fn).reshape((bs,1))
        de=ops.power_scalar(variance+self.eps,0.5).broadcast_to(x.shape)
        return self.weight.reshape((1,fn)).broadcast_to(x.shape)*delta/de+self.bias.reshape((1,fn)).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          mask=init.randb(*x.shape,p=1-self.p,dtype="float32",device=x.device)/(1-self.p)
          return x*mask
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION
class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


