import os
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import re
#####调试用####
import sys
from torchvision.transforms.transforms import Compose
from tqdm import tqdm
#############
class TrainDataset(Dataset):
    def __init__(self,txt_file_root_path,img_file_root_path,sidelen=512,reload_bool=False):
        super(TrainDataset,self).__init__()
        
        # with open(txt_file_path,'r') as f:
        #     self.txt_file_length=len(f.read().splitlines())
        
        self.txt_file_root_path=txt_file_root_path
        self.imgs_root_path=img_file_root_path
        self.txt_file_path=os.path.join(self.imgs_root_path,'vorts0008_infos.txt')
        self.imgs_name=self.__CollectFilePath()
        self.data_length=len(self.imgs_name)
        ###Data_Loading###
        #根据图像名称的列表制作相对应的csv数据，保存在本地（reload_bool=False或者本地已经有可用文件时不进行制作）
        print("InputData Loading...")
        for index,name in enumerate(tqdm(self.imgs_name)):#self.input_data_FilePath是保存为每个csv文件保存的路径
            self.input_data_FilePath=os.path.join(self.txt_file_root_path,name[:-3]+'txt')
            if not os.path.exists(self.input_data_FilePath) or reload_bool:
                self.__CreateInputData(sidelen,index)
        print("Loading finished.")
        self.img_path_buffer=None#Space Saving
        self.img_buffer=None

        # self.data=pd.read_csv(self.input_data_FilePath)

        self.transformer=transforms.Compose([
            transforms.ToTensor()
            ])

    def __getitem__(self,idx):
        """output data & label by processing self.data()

        Args:
            idx (int): index

        Returns:
            img: img values in tensor form
            input_seq: input sequence(one img)
        """
        img_path=os.path.join(self.imgs_root_path,self.imgs_name[idx])
        #---------------get img-----------------#
        img=Image.open(img_path)
        #-----------------Transform---------------------#
        img=self.transformer(img)
        input_seq=np.fromfile(os.path.join(self.txt_file_root_path,self.imgs_name[idx][:-3]+'txt'),sep=' ')
        input_seq=torch.tensor(input_seq.reshape((512,512,5)))#!reshape args here!!!!!
        return img,input_seq

    def __len__(self):
        return self.data_length

    # def __CheckDataNumMatch(self):
    #     with open(self.input_data_FilePath) as f:
    #         dataNum=len(f.read().splitlines())
    #     if self.data_length==dataNum:
    #         return True
    #     else:
    #         return False

    def __CollectFilePath(self, rename=False):
        """返回一个指定目录下(self.img_root_path)所有的png图像文件路径的列表

        Args:
            rename (bool, optional): [如果此参数为True，则重新命名所有的图像文件，保证所有图像文件名称为长度相等的统一格式]. Defaults to False.

        Returns:
            [list]: [一个含有所有图像路径的列表]
        """
        imgs_name=[]
        parten = re.compile(r'[0-9]{4}')
        if rename:
            for root, dirs_name, files_name in os.walk(self.imgs_root_path):
                for file_name in files_name:
                    if file_name[-3:] == 'txt':
                        continue
                    else:
                        if parten.search(file_name[9:]) == None:
                            new_name = 'vorts0008_render_0' + file_name[-7:]
                            new_path = os.path.join(root, new_name)
                            old_path = os.path.join(root, file_name)
                            os.rename(old_path, new_path)

        for root, dirs_name, files_name in os.walk(self.imgs_root_path):
            for file_name in files_name:
                if file_name[-3:] == 'txt':
                    continue
                else:
                    imgs_name.append(file_name)
        return imgs_name

    def __CreateInputData(self,sidelen,index):
        def get_mgrid(sidelen=sidelen, dim=2):
            '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
            sidelen: int
            dim: int
            '''#!此处x,y顺序处理可能有问题，注意检查
            tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
            mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
            mgrid = mgrid.reshape(-1, dim)
            return mgrid

        mgrid=get_mgrid()

        with open(self.input_data_FilePath,'w') as f_write,open(self.txt_file_path,'r') as f_read:
            #In Loop Form
            for position,line in enumerate(f_read):
                if position==index:
                    txt_data=list(map(lambda x:float(x),line.strip().split()))
                    break
            
            """
            The form of one Line:
            img_name y,x txt_value1 txt_value2 txt_value3 y_in x_in
            (where x,y in [0,sidelen] and x_in,y_in in[-1,1])
            """
            cnt=0
            for x in range(sidelen):
                for y in range(sidelen):
                    f_write.write(f'{txt_data[0]} {txt_data[1]} {txt_data[2]} {mgrid[cnt][1]} {mgrid[cnt][0]}\n')
                    cnt+=1
                        
        

if __name__ == "__main__":
    txt_file_root_path='E:/VScodelib/FCNet/data/input_data'
    img_file_root_path='E:/VScodelib/FCNet/data/tiny_vorts0008_normalize_dataset'

    t=DataLoader(TrainDataset(txt_file_root_path,img_file_root_path))
    for data in t:
        label,input_data=data
        print(label.shape)
        print(input_data.shape)
        break
