"""Optimization module"""
import needle as ndl
import numpy as np
import math

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
          if p not in self.u:
            self.u[p]=ndl.init.zeros(*p.shape,device=p.device)
          self.u[p].data=self.momentum*self.u[p].data+(1-self.momentum)*(p.grad.data+self.weight_decay*p.data)
          p.data=p.data-self.lr*self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t+=1
        for param in self.params:
          if param not in self.m:
            self.m[param]=ndl.init.zeros(*param.shape,device=param.device)
          if param not in self.v:
            self.v[param]=ndl.init.zeros(*param.shape,device=param.device)
          tgrad=param.grad.data+self.weight_decay*param.data
          self.m[param].data=self.m[param].data*self.beta1+(1-self.beta1)*tgrad
          self.v[param].data=self.v[param].data*self.beta2+(1-self.beta2)*tgrad*tgrad
          cor_m=self.m[param].data/(1-self.beta1**self.t)
          cor_v=self.v[param].data/(1-self.beta2**self.t)
          param.data=param.data-self.lr*cor_m/(self.eps+cor_v**0.5)
        ### END YOUR SOLUTION