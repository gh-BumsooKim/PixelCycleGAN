import torch
from PIL import Image

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
        
        out = {'pixel_img'      : self.transform(img_p),
               'cartoon_img'    : self.transform(img_c)}
        
        return out
    
    
class CustomDatasetLoader():
    def __init__(self, opt):
        
        self.opt = opt
        self.dataset    = PixelCartoonDataset(opt)
        print("dataset [%s] was created" % type(self.dataset).__class__.__name__)
        
        self.dataloader = torch.utils.data.Dataloader(
            self.dataset,
            batch_size  = self.opt.batch_size,
            shuffle     = opt.batch_shuffle,
            num_workers = int(opt.num_threads),
            drop_last   = True
            )
    
    def load_data(self):
        return self
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        # FIXME
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
                #raise StopIteration
                
            yield data
            
            
def create_dataset(opt) :
    data_loader = CustomDatasetLoader(opt)
    dataset = data_loader.load_data()
    
    return dataset