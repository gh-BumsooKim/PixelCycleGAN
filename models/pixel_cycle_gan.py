import torch
import torch.nn as nn

import itertools

from utils.image_pool import ImagePool
from models.base_model import BaseModel 
from models import network

class PixelCycleGAN(BaseModel):
    def __init__(self, opt):
        
        BaseModel.__init__(self, opt)
    
        self.isTrain = opt.isTrain
    
        # Define Generator
        self.netG_cp = network.define_G(opt)
        self.netG_pc = network.define_G(opt)
        
        # Training
        if self.isTrain:
            self.model_names = ['netG_cp', 'netG_pc', 'netD_c', 'netD_p']
            
            
            # Define Discriminator
            self.netD_c  = network.define_D(opt)
            self.netD_p  = network.define_D(opt)
            
            self.fake_P_pool = ImagePool(opt.pool_size)
            self.fake_C_pool = ImagePool(opt.pool_size)
            
            # Define Criterion
            self.criterionGAN   = network.GANloss()
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt   = nn.L1Loss()
            self.criterionTOP   = network.TOPLoss() 
            
            # Define Loss Params
            self.lambda_sc  = opt.lambda_sc
            self.mu         = opt.mu
            self.alpha      = opt.alpha
            
            # Optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_cp.parameters(), self.netG_pc.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_c.parameters(), self.netD_p.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def set_input(self, input):
        self.real_P = input['pixel_img'].to(self.device)
        self.real_C = input['cartoon_img'].to(self.device)
        
    
    def forward(self):
        # C -> G_cp(C) -> G_pc(G_cp(C))
        # C -> P'      -> C''
        self.fake_p = self.netG_cp(self.real_C)
        self.rec_p  = self.netG_pc(self.fake_p)
        
        # P -> G_pc(P) -> G_cp(G_pc(P))
        # P -> c'      -> p''
        self.fake_c = self.netG_pc(self.real_P)
        self.rec_p  = self.netG_cp(self.fake_c)
        
        
    def backward_G(self):
        ## a) 'Adversarial Loss'
        # (forward)  : D_p(G_cp(C))
        self.loss_G_p = self.criterionGAN(self.netD_p(self.fake_p), True)
        # (backward) : D_c(G_pc(P))
        self.loss_G_c = self.criterionGAN(self.netD_c(self.fake_c), True)
        
        
        ## b) 'Identity Loss'
        # (forward)  : || G_cp(P) - P ||
        self.loss_idt_G_p   = self.criterionIdt(self.netG_cp(self.real_P), self.real_P)
        # (backward) : || G_pc(C) - C ||
        self.loss_idt_G_c   = self.criterionIdt(self.netG_pc(self.real_C), self.real_C)
        self.loss_idt = self.loss_idt_G_p + self.loss_idt_G_c
        
        
        ## c) 'Structural combined Loss'
        # c-1) 'Topology-aware loss'
        self.loss_top = self.criterionTOP()
        # c-2) 'Cylcle consistency loss'
        self.loss_cycle_p = self.criterionCycle(self.rec_p, self.real_P)
        self.loss_cycle_c = self.criterionCycle(self.rec_c, self.real_C)
        self.loss_cycle   = self.loss_cycle_p + self.loss_cycle_c
        self.loss_sc = self.cycle + self.mu*self.loss_top
        
        
        # Total Loss
        self.loss_G = self.loss_G_p + \
                      self.loss_G_c + \
                      self.loss_sc * self.lambda_sc + \
                      self.loss_idt* self.alpha
        
        # Calculate Gradient
        self.loss_G.backward()
        
    def backward_D_basic(self, netD, real, fake):
        
        # Real
        pred_real   = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake   = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
        
    def backward_D_c(self):
        fake_p = self.fake_P_pool.query(self.fake_p)
        self.loss_D_c = self.backward_D_basic(self.netD_c, self.real_P, self.fake_P)
        
    def backward_D_p(self):
        fake_c = self.fake_C_pool.query(self.fake_c)
        self.loss_D_p = self.backward_D_basic(self.netD_p, self.real_C, self.fake_C)
            
    def optimize_parameters(self):
        # forward
        self.forward()
        self.set_requires_grad([self.netD_c, self.netD_p], False)
        
        # Optimize [G_cp and G_pc]
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # Optimize [D_c, D_p]
        self.set_requires_grad([self.netD_c, self.netD_p], True)
        self.optimizer_D.zero_grad()
        self.backward_D_c()
        self.backward_D_p()
        self.optimizer_D.step()
        
        