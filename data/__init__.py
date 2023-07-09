import torch

from data.custom_dataset import PixelCartoonDataset

class CustomDatasetLoader():
    def __init__(self, opt):
        
        self.opt = opt
        self.dataset    = PixelCartoonDataset(opt)
        print("\n### Dataset ###\n%s was created" % self.dataset.__class__.__name__)
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size  = self.opt.batch_size,
            shuffle     = opt.batch_shuffle,
            num_workers = int(opt.num_workers),
            drop_last   = True
            )
    
    def load_data(self):
        return self
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        # FIXME
        for i, data in enumerate(self.dataloader):
            # if i * self.opt.batch_size >= self.opt.max_dataset_size:
            #     break
            #     #raise StopIteration
                
            yield data
            
            
def create_dataset(opt) :
    data_loader = CustomDatasetLoader(opt)
    dataset = data_loader.load_data()
    
    return dataset