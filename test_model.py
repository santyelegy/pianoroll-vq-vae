from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from model import Model



batch_size = 256
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3
# midi data : 128*128
# didn't need to be square, also can be 128*256 or other size
# size 似乎需要是4的倍数
input=torch.zeros([batch_size,1,128,648])

model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

optimizer.zero_grad()
print("begin")
vq_loss, data_recon, perplexity = model(input)
recon_error = F.mse_loss(data_recon, input)
loss = recon_error + vq_loss
loss.backward()

optimizer.step()
print("finish")
a=model.encode(input)
#torch.Size([256, 64, 32, 162])
#batch size, num_hidden , num_residual_hidden , 162 
print(model.encode(input).size())
print(model.decode(a).size())