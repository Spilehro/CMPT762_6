"""Headers"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tryModel import BaseNet
from dataloader import CIFAR100_SFU_CV
import time
# save numpy array as csv file



np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)

# <<TODO#5>> Based on the val set performance, decide how many
# epochs are apt for your model.
# ---------
EPOCHS = 100
# ---------

IS_GPU = True
TEST_BS = 256
TOTAL_CLASSES = 100
PATH_TO_CIFAR100_SFU_CV = "../data/cifar100"
plot_root = 'plots/'
checkpoints_root = 'checkPoints'

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.0,),(1.0,))])
# ---------------------


testset = CIFAR100_SFU_CV(root=PATH_TO_CIFAR100_SFU_CV, fold="test",
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BS,
                                         shuffle=False, num_workers=2)
print("Test set size: "+str(len(testset)))

# The 100 classes for CIFAR100
classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


criterion = nn.CrossEntropyLoss()
net = BaseNet()
if IS_GPU:
    net = net.cuda()

model_name = 'checkPoints/model_best.pth'
net.load_state_dict(torch.load(model_name))
net.eval()
start = time.time()

running_loss = 0.0

count = 0
with open("submission55.csv","w") as file_csv: 
    file_csv.write("ID,Prediction1\n")   
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data

        if IS_GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # forward + backward + optimize
        outputs = net(inputs)
        predictions = torch.softmax(outputs,dim=1)
        _,predicted = torch.max(predictions,dim=1)
        predicted =predicted.detach().cpu().numpy()
        for j in range(len(predicted)):
            file_csv.write(str(count)+','+str(predicted[j])+'\n')
            count+=1
        
    
    
    # Normalizing the loss by the total number of train batches

    
print(time.time()-start)

