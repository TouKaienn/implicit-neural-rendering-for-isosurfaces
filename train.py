from model import *
from train_dataset import *
import matplotlib.pyplot as plt
import numpy as np
from model import *
from train_dataset import *
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', type=str, default=".\\tiny_vorts0008_normalize_dataset\\vorts0008_render_001.png")
    parser.add_argument('--input_txt', type=str,  default=".\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt")
    parser.add_argument('--output_shape', type = int, default=256) # the paper uses 256 for this one
    parser.add_argument('--other_dim', type=int, default=3)
    config = parser.parse_args()

    data = TrainDataset(text_path='.\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt',img_root_path='.\\tiny_vorts0008_normalize_dataset\\')

    

    dataloader = DataLoader(data, batch_size=1, pin_memory=True, num_workers=0)

    net = Siren(in_features=5, out_features=1, hidden_features=50, 
                      hidden_layers=3, outermost_linear=True)

    net.to(device)

    ### train the model
    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=net.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)

    for step in range(total_steps):

        model_output, coords = net(model_input)

        #print("model_output shape is", model_output.size())
        #print("ground truth shape is", ground_truth.size())

        loss = ((model_output - ground_truth) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()