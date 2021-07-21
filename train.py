from model import *
from train_dataset import *
import matplotlib.pyplot as plt
import numpy as np
from model import *
from train_dataset import *
import argparse
import os
from tqdm import tqdm
from torch.nn import MSELoss
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train():
    def __init__(self,config):
        self.epochs=10
        self.time=time.time()
        #############################
        data = TrainDataset(text_path='.\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt',img_root_path='.\\tiny_vorts0008_normalize_dataset')
        self.dataloader = DataLoader(data, batch_size=1, pin_memory=True, num_workers=0)
        self.net = Siren(in_features=5, out_features=1, hidden_features=50, 
                        hidden_layers=3, outermost_linear=True)
        self.optimizer=torch.optim.Adam(lr=1e-4,params=self.net.parameters())
        self.loss_fuc=MSELoss()
        # self.net.to(device=device)
        
    def train(self):
        print("Start Training:")
        for epoch in tqdm(range(self.epochs)):
            
            self.epoch=epoch
            for _, data in enumerate(self.dataloader):
                txt_data,label_img=data
                height,width = label_img.shape[1],label_img.shape[2]
                self.train_one_img(label_img,txt_data,width,height)





    def train_one_img(self,label_img,txt_data,width,height):
        for x in range(width):
            for y in range(height):
                input_data=torch.cat((*txt_data,torch.tensor([x]),torch.tensor([y])))
                ground_truth=label_img[0][y][x]

                # input_data.to(device=device)
                # ground_truth.to(device=device)

                output,_=self.net(input_data)
 
                loss=self.loss_fuc(output.to(torch.float32),ground_truth.to(torch.float32))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.setp()
    
    def write_log(self,loss):
        with open('log.txt','a') as f:
            f.write(f'epoch:{self.epoch}, loss:{loss},time_consuming:{time.time()-self.time}s')
        self.time=time.time()

                

        







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', type=str, default=".\\tiny_vorts0008_normalize_dataset\\vorts0008_render_001.png")
    parser.add_argument('--input_txt', type=str,  default=".\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt")
    parser.add_argument('--output_shape', type = int, default=256) # the paper uses 256 for this one
    parser.add_argument('--other_dim', type=int, default=3)
    config = parser.parse_args()

    train=Train(config)
    train.train()
    

