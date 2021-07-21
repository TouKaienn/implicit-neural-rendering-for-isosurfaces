from model import *
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from torch.nn import MSELoss
class TrainDataset(Dataset):
    def __init__(self, img_root_path, text_path):
        self.img_root_path=img_root_path
        self.text_path=text_path
        self.img_path=self.__CollectFilePath()
        self.all_txt_data=self.__GetTextValue()
        


    def __getitem__(self,idx):
        #read img:
        label_img=np.array(Image.open(self.img_path[idx]))
        #return input label
        return self.all_txt_data[idx],label_img       

    def __len__(self):
        return len(self.all_txt_data)

    def __GetTextValue(self):
        res=[]
        with open(self.text_path,'r') as f:
            for line in f:
                res.append(list(map(float,line.strip().split(' '))))
        return res



    def __CollectFilePath(self,rename=False):
        res=[]
        parten=re.compile(r'[0-9]{4}')
        if rename:
            for root,dirs_name,files_name in os.walk(self.img_root_path):
                for file_name in files_name:
                    if file_name[-3:]=='txt':
                        continue
                    else:
                        if parten.search(file_name[9:])==None:
                            new_name='vorts0008_render_0'+file_name[-7:]
                            new_path=os.path.join(root,new_name)
                            old_path=os.path.join(root,file_name)
                            os.rename(old_path,new_path)

        for root,dirs_name,files_name in os.walk(self.img_root_path):
            for file_name in files_name:
                if file_name[-3:]=='txt':
                    continue
                else:
                    res.append(os.path.join(root,file_name))
        return res



if __name__ == "__main__":
    img_root_path='E:\\VScodelib\\implicit-neural-rendering-for-isosurfaces\\tiny_vorts0008_normalize_dataset'
    text_path='E:\\VScodelib\\implicit-neural-rendering-for-isosurfaces\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt'
    dataset=TrainDataset(img_root_path='E:\\VScodelib\\implicit-neural-rendering-for-isosurfaces\\tiny_vorts0008_normalize_dataset',text_path='E:\\VScodelib\\implicit-neural-rendering-for-isosurfaces\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt')
    data=DataLoader(dataset,batch_size=1)
    for index,item in enumerate(data):
        input_data,label_data=item
        loss=MSELoss()
        print(input_data)
        print(loss(label_data[0][1][0].float(),label_data[0][1][1].float()))
        print(label_data.shape)
        break



                

