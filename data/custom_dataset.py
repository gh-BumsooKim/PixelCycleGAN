import torch
from PIL import Image
#import Image

import glob
from typing import Dict

from data.base_dataset import BaseDataset, get_transform


class PixelCartoonDataset(BaseDataset):
    def __init__(self,
                 opt,
                 transform=None):
        
        self.pixel_img      = glob.glob(opt.path_pixel_dataset)
        self.cartoon_img    = glob.glob(opt.path_cartoon_dataset)
        
        if transform == None:
            self.transform = get_transform(opt)
        else:
            self.transform = transform
        
    def __len__(self):
        return min(len(self.pixel_img), len(self.cartoon_img))
    
    def __getitem__(self, idx) -> Dict[torch.Tensor, torch.Tensor]:
        img_p_path      = self.pixel_img[idx]
        img_c_path      = self.cartoon_img[idx]
        
        img_p, img_c    = Image.open(img_p_path), Image.open(img_c_path)
        img_p, img_c    = img_p.convert('RGB'), img_c.convert('RGB')
        
        out = {'pixel_img_path'     : img_p_path,
               'cartoon_img_path'   : img_c_path,
               'pixel_img'          : self.transform(img_p),
               'cartoon_img'        : self.transform(img_c)}
        
        return out
