import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a=gain*math.sqrt((6/(fan_in+fan_out)))
    return rand(fan_in,fan_out,low=-1*a,high=a,**kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    stdq=gain*math.sqrt((2/(fan_in+fan_out)))
    return randn(fan_in,fan_out,mean=0.0,std=stdq,**kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu",shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound=math.sqrt(6/fan_in)
    if shape is None:
        return rand(fan_in,fan_out,low=-1*bound,high=bound,**kwargs)
    return rand(*shape,low=-1*bound,high=bound,**kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    stdq=math.sqrt(2)/math.sqrt(fan_in)
    return randn(fan_in,fan_out,mean=0.0,std=stdq,**kwargs)
    ### END YOUR SOLUTION