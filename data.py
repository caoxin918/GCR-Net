import numpy as np
import torch
from torch.utils.data import Dataset


class CltDataset(Dataset):
    def __init__(self):
        x = np.load(r"the path of your training data")
        y = np.load(r"the path of your training data")
        self.x_data = torch.from_numpy(x[0:4000,:,:]).to(torch.float32)
        self.y_data = torch.from_numpy(y[0:4000,:]).to(torch.float32)
        self.len = len(self.x_data)

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len


class CltValidateDataset(Dataset):
    def __init__(self):
        x = np.load(r"the path of your validating data")
        y = np.load(r"the path of your validating data")
        self.x_data = torch.from_numpy(x[0:1000,:,:]).to(torch.float32)
        self.y_data = torch.from_numpy(y[0:1000,:]).to(torch.float32)
        self.len = len(self.x_data)

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
