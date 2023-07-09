import torch
import torchvision

from abc import ABC, abstractmethod

import os

class BaseModel(ABC):
    def __init__(self, opt):
        self.model_names = []
        self.optimizers = []
    
    @abstractmethod
    def set_input(self, input: dict):
        """
        Parameters
        ----------
        input : dict
            includes the image itself and its path.

        """
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def optimize_parameters(self):
        pass
    
    def test(self):
        with torch.no_grad(): 
            self.forward()
            
    def save_networks(self, epoch):
        if not os.path.isdir(self.save_dir): os.mkdir(self.save_dir)
        
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%6d_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                #if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                #    torch.save(net.module.cpu().state_dict(), save_path)
                #    net.cuda(self.gpu_ids[0])
                #else:
                #    torch.save(net.cpu().state_dict(), save_path)
                
                torch.save(net.cpu().state_dict(), save_path)
    
    def save_output(self, iters):
        if not os.path.isdir(self.save_output): os.mkdir(self.save_output)
        
        rP = torchvision.utils.make_grid(self.real_P)
        rC = torchvision.utils.make_grid(self.real_C)
        fp = torchvision.utils.make_grid(self.fake_p)
        rp = torchvision.utils.make_grid(self.rec_p)
        fc = torchvision.utils.make_grid(self.fake_c)
        rc = torchvision.utils.make_grid(self.rec_c)
        
        to_save = torch.concat([rP, rC, fp, rp, fc, rc], dim=1)
        save_filename = '%6d.png' % (iters)
        save_path = os.path.join(self.save_output, save_filename)
        torchvision.utils.save_image(to_save, save_path)
        
        return None
    
    def load_networks(self, epoch):
        
        pass
    
    def print_networks(self, verbose):
        print('----------  Networks initialized --------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                n_params = 0
                for param in net.parameters():
                    n_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total n_params : %.3f M' % (name, n_params / 1e6))
                
        
        print('-----------------------------------------------')
    
    def set_requires_gard(self, net, requires_grad=False):
        if not isinstance(net, list):
            nets = [net]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
