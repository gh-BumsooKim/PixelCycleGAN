import torch
import torchvision.transforms as transforms

import glob

from abc import ABC, abstractmethod

class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        
    @abstractmethod
    def __len__(self):
        return 0
    
    @abstractmethod
    def __getitem__(self, index):
        pass
    
    
def get_transform(opt):
    transform_list = [transforms.ToTensor()]
    
    transform_list.append(transforms.Resize([opt.load_size, opt.load_size]))
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
    return transforms.Compose(transform_list)