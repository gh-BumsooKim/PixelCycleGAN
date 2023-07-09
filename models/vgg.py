import torch
import torch.nn as nn

import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, n_layer):
        super().__init__()
        
        vgg16 = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        self.relu1 = nn.Sequential(*vgg16.features[:2])
        self.relu2 = nn.Sequential(*vgg16.features[2:7])
        self.relu3 = nn.Sequential(*vgg16.features[7:12])
        self.relu4 = nn.Sequential(*vgg16.features[12:19])
        self.relu5 = nn.Sequential(*vgg16.features[19:26])
        
        self.layer = []
        for i in range(1, n_layer+1):
            if i == 1: self.layer += [self.relu1]
            if i == 2: self.layer += [self.relu2]
            if i == 3: self.layer += [self.relu3]
            if i == 4: self.layer += [self.relu4]
            if i == 5: self.layer += [self.relu5]
        
        self.n_layer = n_layer

    def forward(self, x):
        out = dict()
        
        for i, layer in enumerate(self.layer):
            x = layer(x)
            out[f'relu{i+1}'] = x
            
        return out 
		
        
        
    