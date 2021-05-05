

from dataset import training_data,validation_data,data_variance
from torch.utils.data import DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES']='4,5'
CUDA_VISIBLE_DEVICES=4

batch_size = 256

training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)
"""
for i in xrange(num_training_updates):
    (data, _) = next(iter(training_loader))
"""
#torch.Size([256, 3, 32, 32])
#I want it to be [batchsize,(channel=1),128,128]
(data, _) = next(iter(training_loader))
print(data.size())