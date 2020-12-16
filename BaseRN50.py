# Import packages
import json
from os import write
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from ResnetModels import ResNet, ResNet50

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
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from Trainer import Trainer
from Hyperparameters import Hyperparameters as Hyp
# Set up trainer
tr = Trainer(ResNet50, 'data', device, batch_size=128)
optim_params = Hyp()
optim_params.register('lr')
optim_params.set_value('lr', 1e-4)
tr.set_hyperparameters(optim_params)
tr.set_hyperparameters(optim_params)
tr.set_criterion(CrossEntropyLoss)
tr.set_optimizer(Adam)
tr.set_scheduler(ReduceLROnPlateau)
tr.prime_optimizer()
#tr.prime_scheduler(milestones=[5], gamma=0.1)
tr.prime_scheduler('max')
tr.prime_model(pretrained=True)
tr.train(epochs=30, save_every=10)
hist = tr.history
write_history('raw', hist)