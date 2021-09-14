# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from model import *
from data.TrainDataset import *
from option import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from tqdm import tqdm
from torch.nn import MSELoss
import time
from datetime import datetime
import numpy as npt
import sys

from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self):
        self.epochs = config.epoch
        self.time = time.time()
        self.batch_size = config.batch_size
        #############################
        
        #if not os.path.exists('.\\net.pkl') or True:
        if not os.path.exists(config.train_weight):
            self.net = Siren()
        else:
            self.load_model()
            print("load model success")

        self.losses=[]
        self.net.to(device=device)
        self.optimizer = torch.optim.Adam(lr=config.lr_rate, params=self.net.parameters())
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.epochs,T_mult=5,eta_min=1e-6)
        # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=20,gamma=0.2,last_epoch=-1)
        self.loss_fuc = MSELoss().to(device)
        self.PSNR = PSNR()
        self.SSIM = SSIM()
        self.loss = None
        self.use_scheduler = config.use_scheduler

        self.data = TrainDataset(config.save_txt_root_path,
                                config.input_image,
                                reload_bool=True,
                                sidelen=config.side_len)
                                
        #self.input_data_FilePath=data.input_data_FilePath
        self.dataloader = DataLoader(self.data, batch_size=self.batch_size,shuffle=True)
        test_training_data_input=self.data.GetTestingData_Testing()
        self.test_training_data_input=test_training_data_input.to(device)
        
        # self.dataloader=train_loader
        self.test_data=TrainDataset('./data/test_data',
                                './data/test_data',
                                reload_bool=True,
                                sidelen=config.side_len)
        self.test_dataloader=DataLoader(self.test_data, batch_size=1,shuffle=False)
        test_testing_data_input=self.test_data.GetTestingData_Testing()
        self.test_testing_data_input=test_testing_data_input.to(device)

        self.test_losses=[]


    def train(self):
        print("Start Training:")
        #self.img_buffer=[]
        plt.figure()
        plt.ion()

        for epoch in tqdm(range(int(self.epochs))):
            # if epoch %100==0:       
            self.epoch = epoch
            for save_count, data in enumerate(self.dataloader):
                label,input_data=data
                label,input_data=self.prepare(label,input_data)#输入是一张图片的采样用来训练
                output,_ = self.net(input_data)
                # print('output.shape:',output.shape)
                # output = output.permute(0,3,1,2)
                # self.psnr = self.PSNR(output, label)
                # self.ssim = self.SSIM(output, label)
                self.loss = self.loss_fuc(output,label)
                # loss=((output-label)**2).mean()
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                #if loss_cpu==min(self.losses):
                    #self.save_model()
                if save_count % 1 == 0:
                    self.test_train_data(epoch)
                    self.save_model(epoch)

            if epoch%10==0:
                test_loss=self.test(epoch)
                
                
            


            loss_cpu=self.loss.cpu().detach()
            self.losses.append(loss_cpu)
            self.write_log(self.loss)

            if test_loss==min(self.test_losses):
                self.save_model(epoch)
            
            plt.clf()
            plt.plot(np.arange(1,len(self.losses)+1,step=1),self.losses)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            # plt.ylim((0,0.1))
            plt.grid()
            plt.pause(0.1)
            
            
            plt.savefig("loss_curve2"+ ".png")
            if self.use_scheduler:
                    self.scheduler.step()
        plt.ioff()

    def test(self,step):
        with torch.no_grad():
            save_dir='.'
            testing_data,_ =self.net(self.test_testing_data_input)#这个是测试的数据集   
            #print('model_output', model_output)
            #print('test_model_output min', test_model_output.min())
            #print('test_model_output max', test_model_output.max())
            ############### option 1
            test_scale_model_ouput = testing_data.cpu().view(512,512,3).detach().numpy() 
            #print('test_scale_model_ouput', test_scale_model_ouput)
            print('\nEpoch:',step)
            print('test_scale_model_ouput min', test_scale_model_ouput.min())
            print('test_scale_model_ouput max', test_scale_model_ouput.max())
            inferenced = test_scale_model_ouput*255
            inferenced = Image.fromarray(inferenced.astype('uint8'))
            direc = save_dir + '/test_iter'
            directory = os.path.join(direc) 
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            flag_func=lambda x:int(x/100)*100+100

            img_save_dir = save_dir + '/test_iter' +f'/epoch_{flag_func(step)}_vorts0008_render_1'+ '.png'
            inferenced.save(img_save_dir)
            #####################################
            loss=0
            for count,data in enumerate(self.dataloader):
                label,input_test_data=data
                label,input_test_data=self.prepare(label,input_test_data)
                output,_=self.net(input_test_data)
                loss+=self.loss_fuc(output,label).cpu().detach()
            
            cur_test_loss=loss/count
            print('test loss:',cur_test_loss)
            self.test_losses.append(cur_test_loss)
            return cur_test_loss



    def test_train_data(self,step):

        #!only one image is enough for test when training
        with torch.no_grad():
            save_dir='.'
            test_model_output, _ = self.net(self.test_training_data_input)#这个是训练的数据集

            testing_data,_ =self.net(self.test_testing_data_input)#这个是测试的数据集   
            #print('model_output', model_output)
            #print('test_model_output min', test_model_output.min())
            #print('test_model_output max', test_model_output.max())
            ############### option 1
            test_scale_model_ouput = test_model_output.cpu().view(512,512,3).detach().numpy() 
            #print('test_scale_model_ouput', test_scale_model_ouput)
            # print('\ntest_scale_model_ouput min', test_scale_model_ouput.min())
            # print('test_scale_model_ouput max', test_scale_model_ouput.max())
            inferenced = test_scale_model_ouput*255
            inferenced = Image.fromarray(inferenced.astype('uint8'))
            direc = save_dir + '/train_iter'
            directory = os.path.join(direc) 
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            flag_func=lambda x:int(x/100)*100+100

            img_save_dir = save_dir + '/train_iter' +f'/epoch_{flag_func(step)}_vorts0008_render_1'+ '.png'
            inferenced.save(img_save_dir)
        
    def write_log(self, loss):
        lr=self.optimizer.param_groups[-1]['lr']
        with open(f'log{datetime.now().strftime("%m%d")}.txt', 'a') as f:
            f.write(
                f'epoch:{self.epoch}, time:{datetime.now().strftime("%m/%d_%H:%M:%S")}, lr:{lr:10f}, loss:{loss:10f} time_consuming:{time.time() - self.time:.2f}s\n')
        self.time = time.time()

    def prepare(self, *args):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if len(args) > 1:
            return (a.float().to(device) for a in args)
    
    def save_model(self, epoch):
        torch.save(self.net,'net.pkl')

    def load_model(self):
        self.net = torch.load(config.train_weight)

