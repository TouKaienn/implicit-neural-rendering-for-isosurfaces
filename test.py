from model import *
import matplotlib.pyplot as plt
import numpy as np
from data.TrainDataset import *
from option import *
import os
from tqdm import tqdm
from torch.nn import MSELoss
import time
# -*- coding: UTF-8 -*-
from datetime import datetime

import numpy as np
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleNet(object):
    pass


class Tester():
    def __init__(self):
        self.time = time.time()
        self.batch_size = config.batch_size
        #############################
        data = TrainDataset(config.save_txt_root_path_test,
                            config.input_image_test,
                            reload_bool=True,
                            sidelen=config.side_len)
        self.dataloader = DataLoader(data, batch_size=self.batch_size)
        self.load_model()
        self.net.to(device=device)
        self.optimizer = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        self.loss_fuc = MSELoss().to(device)
        self.loss = None

    def prepare(self, *args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if len(args) > 1:
            return (a.float().to(device) for a in args)

    def load_model(self):
        self.net = torch.load('.\\net.pkl')

    def test(self):
        print("Start Testing:")
        for _, data in enumerate(self.dataloader):
            label, input_data = data

            label, input_data = self.prepare(label, input_data)

            output, _ = self.net(input_data)
            output = output.permute(0, 3, 1, 2)
            self.visualize_test(output, label)
            self.loss = self.loss_fuc(output, label)
            print("the MSE loss is", self.loss)

    def visualize_test(self, output, label):
        fig, axes = plt.subplots(1, 2, figsize=(6, 6))
        output = output[0].permute(1, 2, 0)
        label = label[0].permute(1, 2, 0)  # permute before show

        axes[0].imshow(output.cpu().detach().numpy())
        axes[0].set_title('Output')
        axes[1].imshow(label.cpu().detach().numpy())
        axes[1].set_title('GT')

        plt.savefig("recent_test" + ".png")




