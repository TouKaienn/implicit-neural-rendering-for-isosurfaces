from model import *
import matplotlib.pyplot as plt
import numpy as np
from model import *
from data.TrainDataset import *
from option import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from tqdm import tqdm
from torch.nn import MSELoss
import time
from datetime import datetime
import numpy as np
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self):
        self.epochs = 1000
        self.time = time.time()
        self.batch_size = config.batch_size
        #############################
        data = TrainDataset(config.save_txt_root_path,
                            config.input_image,reload_bool=True,sidelen=512)
        #self.input_data_FilePath=data.input_data_FilePath
        self.dataloader = DataLoader(data, batch_size=self.batch_size)
        #if not os.path.exists('.\\net.pkl') or True:
        if not os.path.exists('.\\net.pkl'):
            self.net = Siren(in_features=81, out_features=3, hidden_features=50,
                             hidden_layers=3, outermost_linear=True)  # in_features changed to 81 after pos_encoding. Original is 5
        else:
            self.load_model()
        self.losses=[]
        self.net.to(device=device)
        self.optimizer = torch.optim.Adam(lr=1e-3, params=self.net.parameters())
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=50,T_mult=5,eta_min=1e-6)
        self.loss_fuc = MSELoss().to(device)
        self.loss = None

    def train(self):
        print("Start Training:")
        #self.img_buffer=[]
        for epoch in tqdm(range(int(self.epochs))):
            self.epoch = epoch
            for save_count, data in enumerate(self.dataloader):
                label,input_data=data

                label,input_data=self.prepare(label,input_data)
                
                output,_=self.net(input_data)
                output = output.permute(0,3,1,2)

                self.loss=self.loss_fuc(output,label)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                loss_cpu=self.loss.cpu().detach()
                self.losses.append(loss_cpu)
                self.visualize(output,label)
                self.write_log(self.loss)

                self.scheduler.step()
                if loss_cpu==min(self.losses):
                    self.save_model()

    def visualize(self, output,label):
        #展现每个batch第一张图像结果
        fig, axes = plt.subplots(1, 2,figsize=(6,6))
        output = output[0].permute(1, 2, 0)
        label = label[0].permute(1, 2, 0) # permute before show

        axes[0].imshow(output.cpu().detach().numpy())
        axes[0].set_title('Output')
        axes[1].imshow(label.cpu().detach().numpy())
        axes[1].set_title('GT')
        
        plt.savefig("recent"+ ".png")
        
        #绘制loss曲线
        plt.figure()
        plt.plot(np.arange(1,len(self.losses)+1,step=1),self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("loss_curve"+ ".png")
        
    def write_log(self, loss):
        lr=self.optimizer.param_groups[-1]['lr']
        with open(f'log{datetime.now().strftime("%m%d")}.txt', 'a') as f:
            f.write(
                f'epoch:{self.epoch}, time:{datetime.now().strftime("%m/%d_%H:%M:%S")}, lr:{lr:5f}, loss:{loss},  '
                f'time_consuming:{time.time() - self.time:.2f}s\n')
        self.time = time.time()

    def prepare(self, *args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if len(args) > 1:
            return (a.float().to(device) for a in args)
    
    def save_model(self):
        torch.save(self.net, '.\\net.pkl')

    def load_model(self):
        self.net = torch.load('.\\net.pkl')

