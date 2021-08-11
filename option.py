# -*- coding: UTF-8 -*-
import argparse

parser = argparse.ArgumentParser()

# This is for training and testing data
parser.add_argument('--save_txt_root_path', type=str,
                    default='D:/Code/SummerResearch/implicit-neural-rendering-for-isosurfaces-main/data/train_data')
parser.add_argument('--save_txt_root_path_test', type = str,
                    default = 'D:/Code/SummerResearch/implicit-neural-rendering-for-isosurfaces-main/data/test_data')
parser.add_argument('--input_image', type=str,
                    default='D:/Code/SummerResearch/implicit-neural-rendering-for-isosurfaces-main/data/train_data')
parser.add_argument('--input_image_test', type=str,
                    default='D:/Code/SummerResearch/implicit-neural-rendering-for-isosurfaces-main/data/test_data')

# This is for training hyperparameters
parser.add_argument('--epoch', type=int, default=100) # training epochs, default is 10000
parser.add_argument('--side_len', type=int, default=512)
parser.add_argument('--batch_size', type = int, default=10)
parser.add_argument('--train_test', type = str, default='train') # choose 'train' or 'test'

# This is for the model
parser.add_argument('--outermost_linear', type = bool, default=True)
parser.add_argument('--use_residual', type = bool, default=True)
parser.add_argument('--in_features', type = int, default=81)
parser.add_argument('--out_features', type = int, default=3)
parser.add_argument('--hidden_features', type = int, default=50)
parser.add_argument('--hidden_layers', type = int, default=3) # This params needs tuning

config = parser.parse_args()
