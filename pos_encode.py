import torch
import torch.nn as nn
import numpy as np
import sys

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.out_dim = in_features + 2 * in_features * self.num_frequencies
    # def get_num_frequencies_nyquist(self, samples):
    #     nyquist_rate = 1 / (2 * (2 * 1 / samples))
    #     return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        #print('coords shape', coords.shape)

        coords = coords.view(coords.shape[0], coords.shape[1], coords.shape[2], -1, self.in_features)  # this has been changed

        #print('coords after view shape', coords.shape)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]
                #print("--------------------------")
                #print('c shape', c.shape)
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                #print('sin shape', sin.shape)
                #print('cos shape', cos.shape)
                #print('coords_pos_enc shape', coords_pos_enc.shape)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), -1)

        return coords_pos_enc.reshape(coords.shape[0], coords.shape[1], coords.shape[2], self.out_dim) # this has been changed


if __name__ == '__main__':
    coords = torch.tensor([[ 1, 2, 3, 4, 5],
                            [ 6, 7, 8, 9, 10]])
    print(coords.size())
    a = coords[..., :3]
    b = coords[..., 3:]
    c = torch.cat((a, b), dim = -1)

    print(c.size())




    '''positional_encoding1 = PosEncodingNeRF(in_features=2, num_frequencies=4)

    coords1 = torch.ones(3, 4, 5, 2)
    print("--------------------------")
    print("result coords size is", positional_encoding1(coords1).size())'''





