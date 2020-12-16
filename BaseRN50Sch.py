# Import packages
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from ResnetModels import ResNet, ResNet50
import json
def write_history(name, history):
    save_path = name + '.json'
    with open(save_path, 'w') as outfile:
        json.dump(history, outfile)
# Choose device
USE_GPU = True

device = None
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from Trainer import Trainer
from Hyperparameters import Hyperparameters as Hyp
# Set up scheduler breakpoints
step_down = [1, 5, 10, 20]
# Set up trainer
optim_params = Hyp()
optim_params.register('lr')
optim_params.set_value('lr', 1e-4)
for ind in range(2):
    for step in step_down:
        optim_params = Hyp()
        optim_params.register('lr')
        optim_params.set_value('lr', 1e-4)
        if ind == 1:
            optim_params.register('weight_decay')
            optim_params.set_value('weight_decay', 3e-2)
        tr = Trainer(ResNet50, 'data', device, batch_size=128)
        tr.set_hyperparameters(optim_params)
        tr.set_criterion(CrossEntropyLoss)
        tr.set_optimizer(Adam)
        tr.set_scheduler(MultiStepLR)
        tr.prime_optimizer()
        tr.prime_scheduler(milestones=[step], gamma=0.1)
        tr.prime_model(pretrained=True)
        tr.train(epochs=30)
        hist = tr.history
        if ind == 1:
            write_history('sch_l2_'+str(step), hist)
        else:
            write_history('sch_'+str(step), hist)