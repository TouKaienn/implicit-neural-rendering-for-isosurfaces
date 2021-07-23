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
from datetime import datetime
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Train():
    def __init__(self, config):
        self.epochs = 1
        self.time = time.time()
        #############################
        data = TrainDataset(text_path='.\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt',
                            img_root_path='.\\tiny_vorts0008_normalize_dataset')
        self.dataloader = DataLoader(data, batch_size=3, pin_memory=True, num_workers=0)
        if not os.path.exists('.\\net.pkl'):
            self.net = Siren(in_features=5, out_features=3, hidden_features=50,
                             hidden_layers=3, outermost_linear=True)
        else:
            self.load_model()
        self.net.to(device=device)
        self.optimizer = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        self.loss_fuc = MSELoss()
        self.loss = None

    def train(self):
        print("Start Training:")
        for epoch in range(self.epochs):

            self.epoch = epoch

            for step, data in enumerate(self.dataloader):
                txt_data, label_img = data
                print("step is", step)
                print("txt_data is", txt_data)
                print("label_img is", label_img.size())
                height, width = label_img.shape[1], label_img.shape[2]

                self.train_one_img(label_img, txt_data, width, height)

    def train_one_img(self, label_img, txt_data, width, height):
        #for x in tqdm(range(width)):
        for x in tqdm(np.linspace(-1, 1, width)): # x = (-1) + (2/width) * N -> N = (x+1)*256/2
            #for y in range(height):
            for y in np.linspace(-1, 1, height): # N = (y+1)*256/2
                input_data = torch.cat((*txt_data,
                                        torch.tensor([x]),
                                        torch.tensor([y]) )) # this one has problem
                #print("the gross text data is", *txt_data)
                #print("The x tensor is", torch.tensor([x]))
                #print("The y tensor is", torch.tensor([y]))
                #print("the input data is", input_data)
                #ground_truth = label_img[0][y][x]
                ground_truth = label_img[0][int((y+1)*256/2)][int((x+1)*256/2)]

                input_data, ground_truth = self.prepare(input_data, ground_truth)

                output, _ = self.net(input_data)

                self.loss = self.loss_fuc(output, ground_truth)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            self.save_model()  # !记得删了
            self.write_log(self.loss)

    def write_log(self, loss):
        with open(f'log{datetime.now().strftime("%m%d")}.txt', 'a') as f:
            f.write(
                f'epoch:{self.epoch}, time:{datetime.now().strftime("%m/%d_%H:%M:%S")}, loss:{loss},  '
                f'time_consuming:{time.time() - self.time:.2f}s\n')
        self.time = time.time()

    def prepare(self, *args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if len(args) > 1:
            return (a.float().to(device) for a in args)

    def save_model(self):
        torch.save(self.net, 'net.pkl')

    def load_model(self):
        self.net = torch.load('.\\net.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', type=str,
                        default=".\\tiny_vorts0008_normalize_dataset\\vorts0008_render_001.png")
    parser.add_argument('--input_txt', type=str, default=".\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt")
    parser.add_argument('--output_shape', type=int, default=256)  # the paper uses 256 for this one
    parser.add_argument('--other_dim', type=int, default=3)
    config = parser.parse_args()

    train = Train(config)
    train.train()

    # print(datetime.now().strftime("%m/%d_%H:%M"))
