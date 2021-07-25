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
from loss_comp import *
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")#ignore the warning message for testing program easily


class Train():
    def __init__(self, config):
        self.epochs = 1
        self.time = time.time()
        self.batch_size = config.batch_size
        #############################
        data = TrainDataset(text_path=config.input_txt,
                            img_root_path=config.input_image)
        self.dataloader = DataLoader(data, batch_size=self.batch_size, pin_memory=True, num_workers=0)

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

            self.epoch_now = epoch

            for _, data in enumerate(self.dataloader):
                txt_data, label_img = data

                # Concat by row, and then reshape txt to have col of 3
                txt_data = torch.cat(txt_data, dim=0).reshape(-1, 3)

                height, width = label_img.shape[1], label_img.shape[2]

                # train a batch of images
                for ind in range(self.batch_size):
                    #print("index is ", ind)
                    self.train_one_img(label_img[ind], txt_data[ind], width, height)

    def train_one_img(self, label_img, txt_data, width, height):

        for x in tqdm(np.linspace(-1, 1, width)): # x = (-1) + (2/width) * N -> N = (x+1)*256/2

            for y in np.linspace(-1, 1, height): # N = (y+1)*256/2

                input_data = torch.cat((txt_data,
                                        torch.tensor([x]),
                                        torch.tensor([y])))

                #print("txt_data is", txt_data)
                print("label_img size is", label_img.size())
                print("label_img is", label_img)
                #print("the input data is", input_data)
                #print("The x tensor is", torch.tensor([x]))
                #print("The y tensor is", torch.tensor([y]))

                #ground_truth = label_img[0][y][x]
                #ground_truth = label_img[0][int((y+1)*256/2)][int((x+1)*256/2)]
                ground_truth = label_img[int((y + 1) * 256 / 2)][int((x + 1) * 256 / 2)]


                input_data, ground_truth = self.prepare(input_data, ground_truth)

                # output here is not image. It is the (R,G,B) value in a specific pixel
                output, coords = self.net(input_data)


                # the MSE loss
                self.loss = self.loss_fuc(output, ground_truth)

                # (CHANGE THIS ONE!!!) append rgb vectors to image
                output_image = []
                output_image.append(output)

                # visualize images
                self.visualize_model(output_image, coords)#TODO:creating and using a test objects, this one need to change a position

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            self.save_model()  # !记得删了
            self.write_log(self.loss)



    def write_log(self, loss):
        with open(f'log{datetime.now().strftime("%m%d")}.txt', 'a') as f:
            f.write(
                f'epoch:{self.epoch_now}, time:{datetime.now().strftime("%m/%d_%H:%M:%S")}, loss:{loss},  '
                f'time_consuming:{time.time() - self.time:.2f}s\n')
        self.time = time.time()

    def prepare(self, *args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if len(args) > 1:
            return (a.float().to(device) for a in args)

    def visualize_model(self, output, coords):

        print("output is", output)
        print("coords is", coords)

        img_grad = gradient(output, coords)
        img_laplacian = laplace(output, coords)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(output.cpu().view(config.output_shape, config.output_shape).detach().numpy())
        axes[1].imshow(img_grad.norm(dim=-1).cpu().view(config.output_shape, config.output_shape).detach().numpy())
        axes[2].imshow(img_laplacian.cpu().view(config.output_shape, config.output_shape).detach().numpy())
        plt.show()
        plt.close()

    def save_model(self):
        torch.save(self.net, 'net.pkl')

    def load_model(self):
        self.net = torch.load('.\\net.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()#TODO: creating a config files to manipulate those configs easily

    parser.add_argument('--input_image', type=str,
                        default='.\\tiny_vorts0008_normalize_dataset')
    parser.add_argument('--input_txt', type=str, default='.\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt')
    parser.add_argument('--output_shape', type=int, default=256)  # the paper uses 256 for this one
    parser.add_argument('--other_dim', type=int, default=3)
    parser.add_argument('--batch_size', type = int, default=2)
    config = parser.parse_args()

    train = Train(config)
    train.train()

    # print(datetime.now().strftime("%m/%d_%H:%M"))
