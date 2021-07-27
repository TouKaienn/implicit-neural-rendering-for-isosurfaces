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
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Train():
    def __init__(self, config):
        self.epochs = 1
        self.time = time.time()
        self.batch_size = config.batch_size
        #############################
        data = TrainDataset(text_path=config.input_txt,
                            img_root_path=config.input_image,
                            side_length=config.output_shape)
        self.dataloader = DataLoader(data, batch_size=self.batch_size)
        if not os.path.exists('.\\net.pkl'):
            self.net = Siren(in_features=5, out_features=3, hidden_features=50,
                             hidden_layers=3, outermost_linear=True)
        else:
            self.load_model()
        self.net.to(device=device)
        self.optimizer = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        self.loss_fuc = MSELoss()
        self.loss = None

    # matrix for x values
    def matrix_x(self, length):
        # create 2d matrix
        matrix = torch.zeros([length, length, 1])

        # init tmp value and step number
        tmp = -1
        step_num = length * length - 1

        # assign value (-1 to 1, by row and then by col)
        for i in range(length):
            for j in range(length):
                matrix[i][j] = tmp
                tmp += 2 / step_num

        # return matrix
        return matrix

    # matrix for y values
    def matrix_y(self, length):
        # create 2d matrix
        matrix = torch.zeros([length, length, 1])

        # init tmp value and step number
        tmp = -1
        step_num = length * length - 1

        # assign value (-1 to 1, by row and then by col)
        for i in range(length):
            for j in range(length):
                matrix[j][i] = tmp
                tmp += 2 / step_num

        # return matrix
        return matrix

    # matrix for x and y values
    def matrix_x_y(self, length):
        return self.concat_matrix_2(self.matrix_x(length), self.matrix_y(length))

    def matrix_single_txt(self, length, v):
        # create 2d matrix
        matrix = torch.zeros([length, length, 1])

        # add v as value
        matrix = torch.add(matrix, v)

        # return matrix
        return matrix

    def matrix_txt(self, length, v1, v2, v3):
        return self.concat_matrix_3(self.matrix_single_txt(length, v1),
                                    self.matrix_single_txt(length, v2),
                                    self.matrix_single_txt(length, v3))

    def concat_matrix_2(self, a, b):
        res = torch.cat((a, b), dim=-1)
        return res

    def concat_matrix_3(self, a, b, c):
        res = torch.cat((a, b, c), dim=-1)
        return res


    def train(self):
        print("Start Training:")
        for epoch in range(self.epochs):

            self.epoch = epoch

            for _, data in enumerate(self.dataloader):
                txt_data, label_img = data

                # Concat by row, and then reshape txt to have col of 3
                txt_data = torch.cat(txt_data, dim=0).reshape(-1, 3)
                print("txt_data is", txt_data)

                height, width = label_img.shape[-1], label_img.shape[-1]
                #print("label image is", label_img.size())
                #print("txt_data.size is", txt_data.size())

                assert height == width
                length = height or width

                # train a batch of images
                for ind in range(self.batch_size):
                    self.train_one_img(label_img[ind], txt_data[ind], length) # ind indicates a specific image inside the batch

    def train_one_img(self, label_img, txt_data, length):

        # obtain three values (isovalue, alpha, beta)
        v1, v2, v3 = txt_data[0], txt_data[1], txt_data[2]
        #print("v1 is", v1)
        #print("v2 is", v2)
        #print("v3 is", v3)

        # concate matrix to obtain input data
        input_data = self.concat_matrix_2(self.matrix_x_y(length), self.matrix_txt(length, v1, v2, v3))
        print("the input data size is", input_data.size())
        #input_data = torch.cat((txt_data,torch.tensor([x]),torch.tensor([y])))

        #print("txt_data is", txt_data)
        #print("label_img size is", label_img.size())
        #print("the input data is", input_data)
        #print("The x tensor is", torch.tensor([x]))
        #print("The y tensor is", torch.tensor([y]))

        #ground_truth = label_img[0][y][x]
        #ground_truth = label_img[0][int((y+1)*256/2)][int((x+1)*256/2)]

        #ground_truth = label_img[int((y + 1) * 256 / 2)][int((x + 1) * 256 / 2)]

        ground_truth = label_img
        print("the GT size is", ground_truth.size())

        input_data, ground_truth = self.prepare(input_data, ground_truth)

        # output here is not image. It is the (R,G,B) value in a specific pixel
        output, _ = self.net(input_data)

        #print("the output is", output)
        print("the output shape is", output.size())


        # the MSE loss
        self.loss = self.loss_fuc(output, ground_truth)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.save_model()  # !记得删了
        self.write_log(self.loss)

    # visualize images
    #self.visualize_model(output_image, coords)





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

    def visualize_model(self, output, coords):

        #img_grad = gradient(output, coords)
        #img_laplacian = laplace(output, coords)
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes.imshow(output.cpu().view(config.output_shape, config.output_shape).detach().numpy())
        #axes[0].imshow(output.cpu().view(config.output_shape, config.output_shape).detach().numpy())
        #axes[1].imshow(img_grad.norm(dim=-1).cpu().view(config.output_shape, config.output_shape).detach().numpy())
        #axes[2].imshow(img_laplacian.cpu().view(config.output_shape, config.output_shape).detach().numpy())
        plt.show()
        plt.close()

    def save_model(self):
        torch.save(self.net, 'net.pkl')

    def load_model(self):
        self.net = torch.load('.\\net.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', type=str,
                        default='.\\tiny_vorts0008_normalize_dataset')
    parser.add_argument('--input_txt', type=str, default='.\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt')
    parser.add_argument('--output_shape', type=int, default=256)  # the paper uses 256 for this one
    parser.add_argument('--other_dim', type=int, default=3)
    parser.add_argument('--batch_size', type = int, default=1)
    config = parser.parse_args()

    train = Train(config)
    train.train()

    # print(datetime.now().strftime("%m/%d_%H:%M"))
