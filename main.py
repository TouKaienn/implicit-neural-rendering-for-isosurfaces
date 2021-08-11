# -*- coding: UTF-8 -*-
from train import *
from option import *
from test import *

if __name__ == "__main__":
    if config.train_test == 'train':
        t=Trainer()
        t.train()
    elif config.train_test == 'test':
        t = Tester()
        t.test()
    else:
        print("invalid choice. Please enter either train or test")
