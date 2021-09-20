# -*- coding: UTF-8 -*-
import argparse

parser = argparse.ArgumentParser()

# This is for training and testing data
parser.add_argument('--save_txt_root_path', type=str,
                    default='./data/train_data')
parser.add_argument('--save_txt_root_path_test', type = str,
                    default = './data/test_data')
parser.add_argument('--input_image', type=str,
                    default='./data/train_data')
parser.add_argument('--input_image_test', type=str,
                    default='./data/test_data')

# This is for training hyperparameters
parser.add_argument('--epoch', type=int, default=501) # training epochs, default is 10000
parser.add_argument('--side_len', type=int, default=512)
parser.add_argument('--batch_size', type = int, default=120000)
parser.add_argument('--lr_rate', type = float, default=5e-5)
parser.add_argument('--train_test', type = str, default='train') # choose 'train' or 'test'
parser.add_argument('--train_weight', type = str, default='net.pkl') # weight file from which to continue training
parser.add_argument('--use_scheduler', type = bool, default=True)

# This is for the model
parser.add_argument('--is_final_linear', type = bool, default=True)
parser.add_argument('--is_final_res', type = bool, default=True)
parser.add_argument('--in_features', type = int, default=5)
parser.add_argument('--out_features', type = int, default=3)
parser.add_argument('--init_features', type = int, default=256) # original is 128
parser.add_argument('--num_res', type = int, default=15) # original is 10
parser.add_argument('--is_pos_encode', type = bool, default=False)
parser.add_argument('--innermost_res', type = bool, default=True)



# This is for testing parameters
parser.add_argument('--test_weight', type = str, default='net.pkl') # weight file name

config = parser.parse_args()
