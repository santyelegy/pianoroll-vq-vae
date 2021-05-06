import torch.utils.data as data
import torch
import os
import pypianoroll
import numpy as np
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LPD_clean(data.Dataset):
    def __init__(self,test):
        dir_list=self.get_dirs()
        self.data=[]
        count=0
        failcount=0
        for path in dir_list:
            count+=1
            if test:
                if not (count%10==(8 or 9)):
                    continue
            else:
                if count%10==(8 or 9):
                    continue
            
            try:
                multitriack=pypianoroll.load(path)
                for track in multitriack:
                    #split the piano roll here
                    track_roll=torch.from_numpy(track.pianoroll)
                    #split it to size 128*648
                    after_split=track_roll.split(648,dim=0)
                    #need to check here
                    for chunk in after_split:
                        if chunk.size(0)==648:
                            self.data.append(chunk.unsqueeze(dim=0))


            except:
                failcount+=1
        print("finish loading,", failcount," files failed to load")

    def __getitem__(self,index):
        track=self.data[index]
        return track.float()

    def get_dirs(self):
        dir_list=[]
        for filepath,dirnames,filenames in os.walk(r'/disk2/11811719/my-vq-vae/data'):
            for filename in filenames:
                dir_list.append(os.path.join(filepath,filename))
        return dir_list

    def __len__(self):
        return len(self.data)
"""
batch_size=128
dataset=LPD_clean(test=False)
#torch.Size([batch_size, 1(channel),128, 128])
loader = DataLoader(dataset,batch_size=batch_size)
data = next(iter(loader))
print(data.size())
#print(len(dataset))
"""