import torch
import torch.nn as nn
import param


### Time-dependent models ###

class FNNt(nn.Module):
    '''
    Feedforward neural network for the coefficient d(t, x)
    Input:  time and position
    Output: coefficient
    
    The input can be an array of (t x), i.e., (dim+1, n_data) tensor.
    Then, the output becomes an array of (d), i.e., (dim, n_data) tensor.
    '''
    def __init__(self):
        super(FNNt, self).__init__()
        self.n_layer = param.n_layer
        tmp = nn.Sequential()
        tmp.add_module("fc", nn.Linear(param.dim+1, param.n_hidden))
        tmp.add_module("relu", nn.ReLU(inplace=True))
        setattr(self, "layer1", tmp)
        for i in range(param.n_layer-1):
            tmp = nn.Sequential()
            tmp.add_module("fc", nn.Linear(param.n_hidden, param.n_hidden))
            tmp.add_module("relu", nn.ReLU(inplace=True))
            setattr(self, "layer%d" % (i+2), tmp)
        self.out = nn.Linear(param.n_hidden, param.dim)
        
    def forward(self, s, correct=1):
        '''
        argument s = (t x)
        correct is for cancelling the constant factor when the short-time TUR is used
        '''
        for i in range(self.n_layer):
            f = getattr(self, "layer%d" % (i+1))
            s = f(s)
        return self.out(s) / correct

    
class FNNKt(nn.Module):
    '''
    Feedforward neural network with kernel function for the coefficient d(t, x)
    Input:  time and position
    Output: coefficient
    
    The input can be an array of (t x), i.e., (dim+1, n_data) tensor.
    Then, the output becomes an array of (d), i.e., (dim, n_data) tensor.
    '''
    def __init__(self):
        super(FNNKt, self).__init__()
        self.n_layer = param.n_layer
        self.n_output = param.n_output
        self.dim = param.dim
        tmp = nn.Sequential()
        tmp.add_module("fc", nn.Linear(param.dim, param.n_hidden))
        tmp.add_module("relu", nn.ReLU(inplace=True))
        setattr(self, "layer1", tmp)
        for i in range(param.n_layer-1):
            tmp = nn.Sequential()
            tmp.add_module("fc", nn.Linear(param.n_hidden, param.n_hidden))
            tmp.add_module("relu", nn.ReLU(inplace=True))
            setattr(self, "layer%d" % (i+2), tmp)
        self.out = nn.Linear(param.n_hidden, param.dim * param.n_output)
        self.out_func_center = nn.Parameter(torch.linspace(param.t_init, param.t_fin, param.n_output).to(param.device))
        self.out_func_width = nn.Parameter((torch.ones(param.n_output) * (param.t_fin - param.t_init)/param.n_output).to(param.device))
        
    def forward(self, s, correct=1):
        '''
        argument s = (t x)
        correct is for cancelling the constant factor when the short-time TUR is used
        '''
        if s.ndim == 3:
            t = s[:, 0, 0]
            x = s[:, :, 1:]
            for i in range(self.n_layer):
                f = getattr(self, "layer%d" % (i+1))
                x = f(x)
            return torch.einsum('ijkl,il->ijk', self.out(x).reshape(len(s), -1, self.dim, self.n_output),
                                torch.exp(-torch.pow((t.reshape(-1,1)-self.out_func_center.reshape(1,-1))/self.out_func_width.reshape(1,-1), 2))) / correct

        elif s.ndim == 2:
            t = s[0, 0]
            x = s[:, 1:]
            for i in range(self.n_layer):
                f = getattr(self, "layer%d" % (i+1))
                x = f(x)
            return torch.matmul(self.out(x).reshape(-1, self.dim, self.n_output),
                                torch.exp(-torch.pow((t-self.out_func_center)/self.out_func_width, 2))) / correct
        elif s.ndim == 1:
            t = s[0]
            x = s[1:]
            for i in range(self.n_layer):
                f = getattr(self, "layer%d" % (i+1))
                x = f(x)
            return torch.matmul(self.out(x).reshape(self.dim, self.n_output),
                                torch.exp(-torch.pow((t-self.out_func_center)/self.out_func_width, 2))) / correct

    
### Time-independent models ###

class FNN(nn.Module):
    '''
    Feedforward neural network for the coefficient d(x)
    Input:  position
    Output: coefficient
    
    The input can be an array of (x), i.e., (dim, n_data) tensor.
    Then, the output becomes an array of (d), i.e., (dim, n_data) tensor.
    '''
    def __init__(self):
        super(FNN, self).__init__()
        self.n_layer = param.n_layer
        tmp = nn.Sequential()
        tmp.add_module("fc", nn.Linear(param.dim, param.n_hidden))
        tmp.add_module("relu", nn.ReLU(inplace=True))
        setattr(self, "layer1", tmp)
        for i in range(param.n_layer-1):
            tmp = nn.Sequential()
            tmp.add_module("fc", nn.Linear(param.n_hidden, param.n_hidden))
            tmp.add_module("relu", nn.ReLU(inplace=True))
            setattr(self, "layer%d" % (i+2), tmp)
        self.out = nn.Linear(param.n_hidden, param.dim)
        
    def forward(self, s, correct=1):
        '''
        argument s = (x)
        correct is for cancelling the constant factor when the short-time TUR is used
        '''
        for i in range(self.n_layer):
            f = getattr(self, "layer%d" % (i+1))
            s = f(s)
        return self.out(s) / correct

