#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:04:14 2018

@author: yusu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:40:12 2018

"""

import torch
from torch.autograd import Variable
import torch.optim as optim
#from utils import mulvt

import time
import random 
import numpy as np
#import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
#from torch.autograd import Variable
import torch.nn.functional as F
from models import MNIST, load_mnist_data,load_model, show_image

class CW(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_onehot_v, c, modifier, TARGETED):
        #print(c.size(),modifier.size())
        loss1 = c*torch.sum(modifier*modifier)
        #output = net(torch.clamp(xi+modifier,0,1))
        output = self.model.predict(xi+modifier)
        real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
        other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
        #print(real,other)
        if TARGETED:
            loss2 = torch.sum(torch.clamp(other - real, min=0))
        else:
            loss2 = torch.sum(torch.clamp(real - other, min=0))
        error = loss2 + loss1 
        return error,loss1,loss2

    def cw(self, input_xi, label_or_target, c, TARGETED=False):
       
        modifier = Variable(torch.zeros(input_xi.size()).cuda(), requires_grad=True)
        yi = label_or_target
        print(yi.size(),self.model.num_classes)
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda())
        optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000
        best_adv = None
        for it in range(1000):
            optimizer.zero_grad()
            error,loss1,loss2 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            self.model.get_gradient(error)
            #error.backward()
            optimizer.step()
            if (it)%500==0:
                print(error.data[0],loss1.data[0],loss2.data[0]) 
            if loss2.data[0]==0:
                if best_loss1 >= loss1.data[0]:
                    best_loss1 = loss1.data[0]
                    best_adv = modifier.clone()    
        if best_adv is None:
            #print(str(c)+'\t'+'None')
            return None
        else:
            return best_adv
 

    def __call__(self, input_xi, label_or_target, TARGETED=False):
        dis_a = []
        c_hi = 1000*torch.ones(input_xi.size()[0],1).cuda()
        c_lo = 0.01*torch.ones(c_hi.size()).cuda()
        while torch.max(c_hi-c_lo) > 1e-1:
            c_mid = (c_hi + c_lo)/2.0
            c_v = Variable(c_mid)
            adv = self.cw(input_xi, label_or_target, c_v, TARGETED)
            if adv is None:
                c_hi = c_mid
                #print(c_mid)
            else:
                dis = torch.norm(adv).data[0]
                dis_a.append(dis)
                print(dis)
                c_lo = c_mid
        return adv   






def attack_mnist(alpha=0.2, beta=0.001, isTarget= False, num_attacks= 100):
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    print("Length of test_set: ", len(test_dataset))
    #dataset = train_dataset

    net = MNIST()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        
    load_model(net, 'models/mnist_gpu.pt')
    #load_model(net, 'models/mnist_cpu.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    def single_attack(image, label, target = None):
       # show_image(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", model.predict(image))
        CAR = CW(model)
        #if target == None:
         #   adversarial = CW(image,label,target)
            #adversarial = attack_untargeted(model,  image, label, alpha = alpha, beta = beta, iterations = 1000)
        #else:
         #   print("Targeted attack: %d" % target)
            
        adversarial = CAR(image, label, target)
        show_image(adversarial.numpy())
        print("Predicted label for adversarial example: ", model.predict(adversarial))
        return torch.norm(adversarial - image)

    print("\n\n Running {} attack on {} random  MNIST test images for alpha= {} beta= {}\n\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta))
    total_distortion = 0.0

    samples = [6312, 6891, 4243, 8377, 7962, 6635, 4970, 7809, 5867, 9559, 3579, 8269, 2282, 4618, 2290, 1554, 4105, 9862, 2408, 5082, 1619, 1209, 5410, 7736, 9172, 1650, 5181, 3351, 9053, 7816, 7254, 8542, 4268, 1021, 8990, 231, 1529, 6535, 19, 8087, 5459, 3997, 5329, 1032, 3131, 9299, 3910, 2335, 8897, 7340, 1495, 5244,8323, 8017, 1787, 4939, 9032, 4770, 2045, 8970, 5452, 8853, 3330, 9883, 8966, 9628, 4713, 7291, 9770, 6307, 5195, 9432, 3967, 4757, 3013, 3103, 3060, 541, 4261, 7808, 1132, 1472, 2134, 634, 1315, 8858, 6411, 8595, 4516, 8550, 3859, 3526]
    #true_labels = [3, 1, 6, 6, 9, 2, 7, 5, 5, 3, 3, 4, 5, 6, 7, 9, 1, 6, 3, 4, 0, 6, 5, 9, 7, 0, 3, 1, 6, 6, 9, 6, 4, 7, 6, 3, 4, 3, 4, 3, 0, 7, 3, 5, 3, 9, 3, 1, 9, 1, 3, 0, 2, 9, 9, 2, 2, 3, 3, 3, 0, 5, 2, 5, 2, 7, 2, 2, 5, 7, 4, 9, 9, 0, 0, 7, 9, 4, 5, 5, 2, 3, 5, 9, 3, 0, 9, 0, 1, 2, 9, 9]
    for idx in samples:
        #idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        #target = None if not isTarget else random.choice(list(range(label)) + list(range(label+1, 10)))
        target = None if not isTarget else (1+label) % 10
        total_distortion += single_attack(image, label, target)
    
    print("Average distortion on random {} images is {}".format(num_attacks, total_distortion/num_attacks))


if __name__ == '__main__':
    timestart = time.time()
    random.seed(0)
    
    attack_mnist(alpha=2, beta=0.005, isTarget= False)
    #attack_cifar10(alpha=5, beta=0.001, isTarget= False)
    #attack_imagenet(arch='resnet50', alpha=10, beta=0.005, isTarget= False)
    #attack_imagenet(arch='vgg19', alpha=0.05, beta=0.001, isTarget= False, num_attacks= 10)

    #attack_mnist(alpha=2, beta=0.005, isTarget= True)
    #attack_cifar10(alpha=5, beta=0.001, isTarget= True)
    #attack_imagenet(arch='resnet50', alpha=10, beta=0.005, isTarget= True)
    #attack_imagenet(arch='vgg19', alpha=0.05, beta=0.001, isTarget= True, num_attacks= 10)

    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))

