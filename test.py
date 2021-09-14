from model import *
import matplotlib.pyplot as plt
import numpy as np
from data.TrainDataset import *
from data.TestDataset import *
from option import *
import os
from PIL import Image
from tqdm import tqdm
from torch.nn import MSELoss
import time
# -*- coding: UTF-8 -*-
from datetime import datetime
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import sys

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Tester():
    def __init__(self):
        self.time = time.time()
        self.batch_size = config.batch_size
        #############################
        data = TestDataset(config.save_txt_root_path_test,
                            config.input_image,
                            reload_bool=True,
                            sidelen=config.side_len)
        self.dataloader = DataLoader(data, batch_size=1)
        self.load_model()
        self.net.to(device=device)
        self.loss_fuc = MSELoss().to(device)
        self.loss = None
        self.transform = Compose([
            ToTensor(),
            ])

    def prepare(self, *args):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        if len(args) > 1:
            return (a.float().to(device) for a in args)
        else:
            return args[0].float().to(device)

    def load_model(self):
        self.net = torch.load(config.test_weight)

    def test(self):
        print("Start Testing:")
        for idx, data in enumerate(self.dataloader):
            input_data,img_path = data
            img_path=img_path[0]
            label=Image.open(img_path).convert('RGB')
            label=self.transform(label)


            input_data,label = self.prepare(input_data,label)


            label=label.view(-1,3)
                
            output,_= self.net(input_data)
            inferenced=output.cpu().view(512,512,3).detach().numpy()
            inferenced=inferenced*255
            inferenced=Image.fromarray(inferenced.astype('uint8'))
            inferenced.save(f'recent_test_{idx}.png')

            # self.visualize_test(output, img_path)
            self.loss = self.loss_fuc(output, label)
            print("the MSE loss is", self.loss)


    # def visualize_test(self, output, img_path):
    #     fig, axes = plt.subplots(1, 2, figsize=(6, 6))
    #     # output = output[0].permute(1, 2, 0) # permute before show
    #     label=Image.open(img_path)
    #     axes[0].imshow(output.cpu().detach().numpy())
    #     axes[0].set_title('Output')
    #     axes[1].imshow(label)
    #     axes[1].set_title('GT')


    #     plt.savefig('recent' + ".png")




