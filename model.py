import torch.nn as nn
from torch.nn import init
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
import torch
from collections import OrderedDict 
import math
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from option import *
import sys
## ref: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=ZnC9GBF8fesa

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies):
        super().__init__()

        self.in_features = in_features

        self.num_frequencies = num_frequencies

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), -1)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(5 * input)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
        
    def forward(self, input):
        return self.linear(input)


class ResBlock(nn.Module):
    def __init__(self,in_features,out_features,nonlinearity='relu'):
        super(ResBlock,self).__init__()
        nls_and_inits = {'sine':Sine(),
                         'relu':nn.ReLU(inplace=True),
                         'sigmoid':nn.Sigmoid(),
                         'tanh':nn.Tanh(),
                         'selu':nn.SELU(inplace=True),
                         'softplus':nn.Softplus(),
                         'elu':nn.ELU(inplace=True)}

        nl = nls_and_inits[nonlinearity]

        self.net = []

        self.net.append(SineLayer(in_features,out_features))

        self.net.append(SineLayer(out_features,out_features))

        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features)
        
        #self.net.append(nn.BatchNorm1d(out_features))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)

    
class Siren(nn.Module):
    def __init__(self,
                 in_features=config.in_features,
                 init_features=config.init_features,
                 hidden_layers=config.num_res,
                 out_features=config.out_features,
                 is_final_linear=config.is_final_linear,
                 is_final_res = config.is_final_res,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 is_pos_encode = config.is_pos_encode,
                 innermost_res = config.innermost_res):
        super().__init__()
        
        self.is_pos_encode = is_pos_encode
        self.hidden_layers = hidden_layers
        self.innermost_res = innermost_res
        self.is_final_linear = is_final_linear
        self.is_final_res = is_final_res

        if self.is_pos_encode:
            self.positional_encoding_xy = PosEncodingNeRF(2, 4) # 4
            self.positional_encoding_values = PosEncodingNeRF(3, 10)  #10

            in_features = self.positional_encoding_xy.out_dim+self.positional_encoding_values.out_dim

        self.net = []

        # First layer
        if self.innermost_res:
            self.net.append(ResBlock(in_features, init_features))
        else:
            self.net.append(SineLayer(in_features, init_features, bias =True,
                                  is_first=True, omega_0=first_omega_0))

        # Subsequent layers
        for i in range(self.hidden_layers):
            self.net.append(ResBlock(init_features, init_features))


        # Final layer
        if self.is_final_linear:
            final_linear = nn.Linear(init_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / init_features) / hidden_omega_0, 
                                              np.sqrt(6 / init_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            if self.is_final_res:
                self.net.append(ResBlock(init_features, out_features))
            else:
                self.net.append(SineLayer(init_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.is_pos_encode:
            coords_values = coords[:,:-2]
            #print('coords_values ', coords_values)
            #print('coords_values shape', coords_values.shape)

            coords_xy = coords[:,-2:]
            #print('coords_xy ', coords_xy)
            #print('coords_xy shape', coords_xy.shape)
            coords_positional_xy = self.positional_encoding_xy(coords_xy)
            #print('coords_positional_xy shape', coords_positional_xy.shape)

            coords_positional_value = self.positional_encoding_values(coords_values)
            #print('coords_positional_value shape', coords_positional_value.shape)

            coords = torch.cat((coords_positional_value,coords_positional_xy), 1)
            #print('coords_positional shape', coords_positional.shape)


        # scale the model from (-1, 1) to (0, 1)
        if self.is_final_linear:
            output = self.net(coords)
            output = torch.sigmoid(output)

        else:
            output = torch.add(torch.div(self.net(coords), 2.0), 0.5)

        return output, coords

    





