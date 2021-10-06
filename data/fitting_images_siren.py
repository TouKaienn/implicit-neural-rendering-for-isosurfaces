import torch.nn as nn
import torch.optim as optim
import time
import argparse
import torch
import numpy as np
# from model import *
# from CoordNet import *
from torch.autograd import Variable
import csv
from math import pi
from PIL import Image
import skimage
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import rotate
from skimage.transform import rescale
import random
import numpy
import os
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


total_steps = 501
img_width = 512
img_height = 512
train_image_numbs = 72
test_image_numbs = 50
#train_images_numbs = 2363 # 0.7*24389
## evry time sample (512*512)/2 pixels of each image
batch_size = 300000 #262144  # the whole image
samples_num = 48000 #48000  #24000

dataset_dir = './data/train_data'
# save_dir = '/afs/crc.nd.edu/user/p/pgu/Research/Isosurface_rendering/isosurfaces_rendering_45resi_connec_72images/result/'
# model_path = '/afs/crc.nd.edu/user/p/pgu/Research/Isosurface_rendering/isosurfaces_rendering_45resi_connec_72images/saved_model/'

test_dataset_dir = './data/train_data'


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    coords_numpy = pixel_coords.numpy()
    return coords_numpy, pixel_coords



def get_pxiels(path, sidelength):
    '''load the image and get the corresponding pixel RGB values that used for GT'''
    img = Image.open(path)
    transform = Compose([
        ToTensor(),
    ])
    img2 = transform(img)
    pixels = img2.permute(1, 2, 0)
    #print('pixels max', pixels.max())
    #print('pixels min', pixels.min())


    #print('pixels', pixels)
    #print('pixels shape', pixels.shape) # torch.Size([512, 512, 3])
    #pixels = pixels.view(sidelength*sidelength, 3)
    #print('pixels view', pixels)
    #print('pixels shape', pixels.shape)
    pixels_view = pixels.view(-1, 3)
    #print('pixels_view ', pixels_view)
    #print('pixels_view shape', pixels_view.shape)
    #print('pixels_view max', pixels_view.max())
    #print('pixels_view min', pixels_view.min())

    #print('pixels_view[56783]', pixels_view[56783])
    #print('pixels[56783]', pixels[56783])
    #print('test pixels', pixels-pixels_view)
    #print('check nonzeros', torch.nonzero(pixels-pixels_view))
    # print(torch.nonzero(torch.tensor([[0.0, 0.0, 0.0, 0.0],
    #                          [0.0, 0.0, 0.0, 0.0],
    #                         [0.0, 0.0, 0.0, 0.0],
    #                          [0.0, 0.0, 0.0,0.0]])))
    return pixels_view


def ReadTrainingData(dataset_dir, train_images_indices):
    '''Read the iso, theta, and phi data from txt
        get the iso, theta, and phi and RGB values for training giving train_images_indices
    '''
    file = open(dataset_dir+'/vorts0008_infos.txt','r')
    Lines = file.readlines()

    isovalue_theta_phi_all = []
    for line in Lines:
        isovalue_theta_phi = []
        for c in line.split():
            #print(float(c))
            isovalue_theta_phi.append(float(c))
        isovalue_theta_phi_all.append(isovalue_theta_phi)
 

    isovalue_theta_phi_input = []
    pixel_ground_truth = []
    for i in train_images_indices:
        #np.array([-0.9333 -1.000 -0.8571])
        isovalue_theta_phi = isovalue_theta_phi_all[i-1]
        #print('isovalue_theta_phi', isovalue_theta_phi)
        isovalue_theta_phi_input.append(isovalue_theta_phi)
        
        
        path = dataset_dir+'/vorts0008_render_' + '{:03d}'.format(i) + '.png'
        #print('path', path)
        pixels_RGB = get_pxiels(path, 512)
        pixels_RGB_numpy = pixels_RGB.numpy()
        pixel_ground_truth.append(pixels_RGB_numpy)


    return isovalue_theta_phi_input, pixel_ground_truth

def GetTrainingData(isovalue_theta_phi_input,pixel_ground_truth,coordinates):
    '''
    constructing the training input and RGB GT values for trianing
    '''
    coords_input = []
    pixels_values = []
    coords_input_test = []
    samples = img_width*img_height
    iso = np.zeros((samples_num,1))
    theta = np.zeros((samples_num,1))
    phi = np.zeros((samples_num,1))
    idx = 0
    for e in isovalue_theta_phi_input:
        #print('e', e)
        #ensembles = [e] * samples_num#samples_num len(indeices)
        #print('ensembles', ensembles)
        #print(ensembles[0])
        #print(ensembles[1])
        index = np.random.randint(low=0,high=samples, size=samples_num)#indeices#np.random.randint(low=0,high=samples, size=samples_num)#batch_size*factor) #[0,512*512-1] #np.random.randint(1,samples, 2)
        #print('index', index)
        #print('index min', index.min())
        #print('index max', index.max())
        #print('coordinates[index]',coordinates[index])


        iso.fill(e[0])
        theta.fill(e[1])
        phi.fill(e[2])

        coords_input +=list(np.concatenate((iso,theta,phi,coordinates[index]),axis=1))#list(coordinates[index])#list(np.concatenate((ensembles,coordinates[index]),axis=1))
        #print('coords_input',coords_input)


        # coords_input +=list(np.concatenate((ensembles,coordinates[index]),axis=1))#list(coordinates[index])#list(np.concatenate((ensembles,coordinates[index]),axis=1))
        # print('coords_input',coords_input)
        
        



        pixels_values += list(pixel_ground_truth[idx][index])
        #print('pixels_values',pixels_values)
        idx +=1

    #print('coords_input',coords_input)
    #print('np.asarray(coords_input)',np.asarray(coords_input))

    # training_data_input_test = torch.FloatTensor(coords_input)
    # print('training_data_input_test',training_data_input_test)

    training_data_input = torch.FloatTensor(np.asarray(coords_input))
    training_data_gt = torch.FloatTensor(np.asarray(pixels_values))
    #print('training_data_input',training_data_input)
    #print('training_data_input shape',training_data_input.shape)
    

    #print('training_data_gt',training_data_gt)
    #print('training_data_gt shape',training_data_gt.shape)
    #print('training_data_gt min',training_data_gt.min())
    #print('training_data_gt max',training_data_gt.max())

    data = torch.utils.data.TensorDataset(training_data_input, training_data_gt)
    train_loader = DataLoader(dataset =data, batch_size = batch_size, shuffle=True)
    return train_loader



def GetTrainingData_Testing(isovalue_theta_phi_input,coordinates):
    '''
    constructing the training input with whole image to fit the training image
    '''
    test_coords_input = []
    samples = img_width*img_height
    for e in isovalue_theta_phi_input:
        #print('e', e)
        ensembles = [e] * samples
        ensembles = torch.from_numpy(np.array(ensembles))
        #print('ensembles', ensembles)
        #print(ensembles[0])
        #print(ensembles[1])
        #index = np.random.randint(low=0,high=img_width*img_height, size=samples)
        #print('torch.FloatTensor(ensembles) ', ensembles.float())
        test_input_ = torch.cat([ensembles.float(),coordinates],1)
        #print('test_coords_input',test_coords_input)
        test_coords_input.append(test_input_)
    #testing_data_input = torch.FloatTensor(test_coords_input)
    return test_coords_input

##########################################
def ReadTestingData(test_dataset_dir, test_images_indices):
    '''Read the iso, theta, and phi data from txt
        get the iso, theta, and phi for testing giving test_images_indices
    '''
    file = open(test_dataset_dir+'/vorts0008_infos.txt','r')
    Lines = file.readlines()

    isovalue_theta_phi_all = []
    for line in Lines:
        isovalue_theta_phi = []
        for c in line.split():
            #print(float(c))
            isovalue_theta_phi.append(float(c))
        isovalue_theta_phi_all.append(isovalue_theta_phi)
 

    isovalue_theta_phi_input_test = []
    for i in test_images_indices:
        #np.array([-0.9333 -1.000 -0.8571])
        isovalue_theta_phi_test = isovalue_theta_phi_all[i-1]
        #print('isovalue_theta_phi', isovalue_theta_phi)
        isovalue_theta_phi_input_test.append(isovalue_theta_phi_test)
       
    return isovalue_theta_phi_input_test

def GetTestingData_Testing(isovalue_theta_phi_input_test,coordinates):
    '''
    constructing the testing input with whole image to test the novel testing image
    '''
    test_coords_input = []
    samples = img_width*img_height
    for e in isovalue_theta_phi_input_test:
        #print('e', e)
        ensembles = [e] * samples
        ensembles = torch.from_numpy(np.array(ensembles))
        #print('ensembles', ensembles)
        #print(ensembles[0])
        #print(ensembles[1])
        #index = np.random.randint(low=0,high=img_width*img_height, size=samples)
        #print('torch.FloatTensor(ensembles) ', ensembles.float())
        test_input_ = torch.cat([ensembles.float(),coordinates],1)
        #print('test_coords_input',test_coords_input)
        test_coords_input.append(test_input_)
    #testing_data_input = torch.FloatTensor(test_coords_input)
    return test_coords_input





# def inference():
#     pass

# if __name__ == '__main__':
# import time
# start_time=time.time()
train_all_indices = list(range(1, train_image_numbs+1))
test_all_indices = list(range(1, test_image_numbs+1))
train_images_indices = train_all_indices
test_images_indices =  test_all_indices

# train_images_indices = [i for i in range(1,73)]#train_all_indices#[12, 43, 52]#random.sample(range(1, 289), 8) #[17] #all_indices #[17] #all_indices
#print(train_images_indices)
test_images_indices = [3]#test_all_indices#[26]

coordinates, coordinates_tensor = get_mgrid(512, dim=2)
# print('coordinates ', coordinates)
# print('coordinates[1] ', coordinates[1])
# print('coordinates[1][0] ', coordinates[1][0])
# print('coordinates[1][1] ', coordinates[1][1])

isovalue_theta_phi_input, pixel_ground_truth = ReadTrainingData(dataset_dir,train_images_indices)
test_training_data_input = GetTrainingData_Testing(isovalue_theta_phi_input, coordinates_tensor)
train_loader = GetTrainingData(isovalue_theta_phi_input,pixel_ground_truth,coordinates)

# end_time=time.time()
# print('time:',end_time-start_time)

















