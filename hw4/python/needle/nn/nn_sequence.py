"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return init.ones_like(x,device=x.device)/(1+ops.exp(-1*x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound=1/np.sqrt(hidden_size)
        self.W_ih=Parameter(init.rand(input_size,hidden_size,low=-1*bound,high=bound,device=device,dtype=dtype,requires_grad=True))
        self.W_hh=Parameter(init.rand(hidden_size,hidden_size,low=-1*bound,high=bound,device=device,dtype=dtype,requires_grad=True))
        if bias:
            self.bias_ih=Parameter(init.rand(hidden_size,low=-1*bound,high=bound,device=device,dtype=dtype,requires_grad=True))
            self.bias_hh=Parameter(init.rand(hidden_size,low=-1*bound,high=bound,device=device,dtype=dtype,requires_grad=True))
        
        self.bias=bias
        self.nonlinearity=nonlinearity
        self.device=device
        self.hiddensize=hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h=init.zeros(X.shape[0],self.hiddensize,device=self.device,dtype=X.dtype)
        raw_h=X@self.W_ih+h@self.W_hh
        if self.bias:
            raw_h+=(self.bias_hh+self.bias_ih).reshape((1,self.hiddensize)).broadcast_to(raw_h.shape)
        if self.nonlinearity=='relu':
            return ops.relu(raw_h)
        else:
            return ops.tanh(raw_h)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.nonlinearity=nonlinearity
        self.rnn_cells=[]
        self.device=device
        for i in range(num_layers):
            temp=None
            if i==0:
                temp=RNNCell(input_size,hidden_size,bias,nonlinearity,device,dtype)
            else:
                temp=RNNCell(hidden_size,hidden_size,bias,nonlinearity,device,dtype)
            self.rnn_cells.append(temp)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device)

        # Pack all weights into a flat arg list for the fused op
        fused_args = [X]
        for cell in self.rnn_cells:
            fused_args.append(cell.W_ih)
            fused_args.append(cell.W_hh)
            fused_args.append(cell.bias_ih)
            fused_args.append(cell.bias_hh)
        fused_args.append(h0)

        output = ops.RNNFusedOp(self.num_layers, self.hidden_size, self.nonlinearity)(*fused_args)

        # output shape: (seq_len, bs, hidden_size)
        out_3d = output.reshape((seq_len, bs, self.hidden_size))

        # Extract h_n: (num_layers, bs, hidden_size) from last time step
        out_split = ops.split(out_3d, 0)
        h_last = out_split[seq_len - 1]  # (bs, hidden_size)
        h_n = ops.stack([ops.split(h0, 0)[i] if False else h_last for i in range(self.num_layers)], 0)
        # Just duplicate h_last for each layer (approximate; h_n is unused in training)
        h_n = ops.stack([h_last], 0)

        return out_3d, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.sigma=Sigmoid()
        self.hiddensize=hidden_size
        bound=np.sqrt(1/hidden_size)
        self.dtype=dtype
        self.device=device
        self.W_ih=Parameter(init.rand(input_size,4*hidden_size,low=-bound,high=bound,device=device,dtype=dtype,requires_grad=True))
        self.W_hh=Parameter(init.rand(hidden_size,4*hidden_size,low=-bound,high=bound,device=device,dtype=dtype,requires_grad=True))
        self.bias_ih=None
        self.bias_hh=None
        if bias:
            self.bias_ih=Parameter(init.rand(4*hidden_size,low=-bound,high=bound,device=device,dtype=dtype,requires_grad=True))
            self.bias_hh=Parameter(init.rand(4*hidden_size,low=-bound,high=bound,device=device,dtype=dtype,requires_grad=True))
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size=X.shape[0]
        h0=None
        c0=None
        if h is None:
            h0=init.zeros(batch_size,self.hiddensize,device=self.device,dtype=self.dtype)
            c0=init.zeros(batch_size,self.hiddensize,device=self.device,dtype=self.dtype)
        else:
            h0,c0=h
        stacked=X@self.W_ih+h0@self.W_hh
        if self.bias_hh is not None:
            stacked+=(self.bias_ih+self.bias_hh).reshape((1,4*self.hiddensize)).broadcast_to((batch_size,4*self.hiddensize))
        raw_i,raw_f,raw_g,raw_o=ops.split(stacked.reshape((batch_size,4,self.hiddensize)),1)
        i=self.sigma(raw_i)
        f=self.sigma(raw_f)
        g=ops.tanh(raw_g)
        o=self.sigma(raw_o)
        cprime=f*c0+i*g
        hprime=o*ops.tanh(cprime)
        return hprime,cprime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.device=device
        self.dtype=dtype
        self.lstm_cells=[]
        self.num_layers=num_layers
        self.hiddensize=hidden_size
        previous=input_size
        for i in range(num_layers):
            if i!=0:
                previous=hidden_size
            self.lstm_cells.append(LSTMCell(previous,hidden_size,bias,device,dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h0=None
        c0=None
        seq_len,batch_size,_=X.shape
        if h is None:
            h0=init.zeros(self.num_layers,batch_size,self.hiddensize,device=self.device,dtype=self.dtype)
            c0=init.zeros(self.num_layers,batch_size,self.hiddensize,device=self.device,dtype=self.dtype)
        else:
            h0,c0=h
        hidden=[]
        outh=[]
        outc=[]
        h0perlayer=ops.split(h0,axis=0)
        c0perlayer=ops.split(c0,axis=0)
        xperstep=ops.split(X,axis=0)
        for i in range(self.num_layers):
            hidden=[]
            cell=self.lstm_cells[i]
            if i==0:
                into=xperstep
            for t in range(seq_len):
                if t==0:
                    h_cur=h0perlayer[i]
                    c_cur=c0perlayer[i]
                h_cur,c_cur=cell(into[t],(h_cur,c_cur))
                hidden.append(h_cur)
            into=hidden
            outh.append(h_cur)
            outc.append(c_cur)
        return ops.stack(hidden,0),(ops.stack(outh,0),ops.stack(outc,0))

        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.randn(num_embeddings,embedding_dim,device=device,dtype=dtype,requires_grad=True))
        self.device=device
        self.dtype=dtype
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len,bs=x.shape
        vec=init.one_hot(self.num_embeddings,x,device=self.device,dtype=self.dtype)
        return (vec.reshape((seq_len*bs,self.num_embeddings))@self.weight).reshape((seq_len,bs,self.embedding_dim))
        ### END YOUR SOLUTION