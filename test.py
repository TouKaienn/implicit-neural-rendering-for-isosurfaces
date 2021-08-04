from model import *
import matplotlib.pyplot as plt
import numpy as np
from model import *
from data.TrainDataset import *
from option import *
import os
from tqdm import tqdm
from torch.nn import MSELoss
import time
from datetime import datetime
import numpy as np
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Test():
    def __init__(self):
        self.epochs = 1
        self.time = time.time()
        self.batch_size = config.batch_size
        #############################
        data = TrainDataset(config.input_txt,
                            config.input_image)
        self.dataloader = DataLoader(data, batch_size=self.batch_size)
        if not os.path.exists('.\\net.pkl'):
            self.net = SimpleNet()
        else:
            self.load_model()
        self.net.to(device=device)
        self.optimizer = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        self.loss_fuc = MSELoss().to(device)
        self.loss = None