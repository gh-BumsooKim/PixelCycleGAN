import torch
import torch.nn as nn
from torch.nn import init

from models.vgg import VGG16

class Sample(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=3,
                 stride=1, padding=1, mode='down', init_layer=False):
        super().__init__()
        
        if mode=='down':
            actv_layer = nn.LeakyReLU()
            final_channels = out_channels
        elif mode=='up':
            actv_layer = nn.ReLU()
            final_channels = out_channels
        elif mode=='out':
            actv_layer = nn.Tanh()
            final_channels = 3
        else:
            raise NotImplementedError()
        
        if init_layer:
            in_channels = 3
        
        self.conv = nn.Sequential(
            actv_layer,
            nn.Conv2d(in_channels,  out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, final_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(final_channels)
            )
        
    def forward(self, x):
        return self.conv(x)
        
        
class Downsample(nn.Module):
    def __init__(self, in_channels=512, factor=2, bilinear=False,
                 init_layer=False):
        super().__init__()
        
        conv = []
        if not init_layer:
            conv.append(nn.MaxPool2d(factor))
            
        conv.append(Sample(mode='down', in_channels=in_channels, init_layer=init_layer))
        
        self.conv = nn.Sequential(*conv)
        
    def forward(self, x):
        return self.conv(x)
    
    
class Upsample(nn.Module):
    def __init__(self, mode='up', in_channels=512, out_channels=512,
                 kernel_size=3, factor=2, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=factor)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        
        self.conv = Sample(mode=mode, in_channels=in_channels)
        
    
    def forward(self, *x_skip, upsample):    
        x_upsampled = self.upsample(upsample)
        x_concat = torch.concat([*x_skip, x_upsampled], dim=1)
        
        out = self.conv(x_concat)
        
        return out
    

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        
        # Downsampling
        downsample_module = [Downsample(factor=2, init_layer=True)]
        for _ in range(opt.n_down):
            downsample_module += [Downsample(factor=2)]
        
        self.downsample = nn.Sequential(*downsample_module)
        
        # Upsampling level-1
        self.upsample1 = nn.Sequential(
            *[Upsample(factor=2, in_channels=512 * 2) for _ in range(opt.n_up1)])
        
        # Upsampling level-2
        self.upsample2 = nn.Sequential(
            *[Upsample(factor=2, in_channels=512 * 3) for _ in range(opt.n_up2)])
        
        # Upsampling level-3
        self.upsample3 = nn.Sequential(
            *[Upsample(factor=2, in_channels=512 * 4) for _ in range(opt.n_up3)])
        
        # Out Layer
        self.out = Upsample(mode='out', in_channels=512 * 5)
        
    
    def forward(self, x):
        
        # Downsampling
        x_downsample_list = []
        for layer in self.downsample:
            x = layer(x)
            x_downsample_list.append(x)
            
        # Upsampling level-1
        x_upsample1_list = []
        for i, layer in enumerate(self.upsample1):
            x_in            = x_downsample_list[i]
            x_to_upsample   = x_downsample_list[i+1]
            
            x_out = layer(x_in, upsample=x_to_upsample)
            
            x_upsample1_list.append(x_out)
            
            
        # Upsampling level-2
        x_upsample2_list = []   
        for i, layer in enumerate(self.upsample2):
            x_in            = [x_downsample_list[i], x_upsample1_list[i]]
            x_to_upsample   = x_upsample1_list[i+1]
            
            x_out = layer(*x_in, upsample=x_to_upsample)
            
            x_upsample2_list.append(x_out)
            
            
        # Upsampling level-3
        x_upsample3_list = []
        for i, layer in enumerate(self.upsample3):
            x_in            = [x_downsample_list[i], x_upsample1_list[i],
                               x_upsample2_list[i]]
            x_to_upsample   = x_upsample2_list[i+1]
            
            x_out = layer(*x_in, upsample=x_to_upsample)
            
            x_upsample3_list.append(x_out)
            
            
        # Out Layer
        x_in            = [x_downsample_list[0], x_upsample1_list[0],
                           x_upsample2_list[0],  x_upsample3_list[0]]
        x_to_upsample   = x_upsample3_list[1]
        x_out = self.out(*x_in, upsample=x_to_upsample)
        
        return x_out
            
    
# class Discriminator(nn.Module):
#     """
#     PatchGAN D Advantage 1: Network Size Reduction (Smaller params)
#     PatchGAN D Advantage 2: Flexibility about input image size by sliding window
#     PatchGAN D Advantage 3: Capture the High-frequency details
    
#     """
#     def __init__(self, opt, in_channels=3, out_channels=64):
#         super().__init__()
        
#         #self.receptive_field = 70
#         if opt.discriminator_norm == "batch_norm":
#             self.norm_layer = nn.BatchNorm2d
#         elif opt.discriminator_norm == "instance_norm":
#             self.norm_layer = nn.InstanceNorm2d
        
        
#         sequence = [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)),
#                     nn.LeakyReLU(0.2, True)]
        
#         for _ in range(1, opt.n_layers):
#             sequence += [
#                 nn.Conv2d(),
#                 self.norm_layer(),
#                 nn.LeakyReLU(0.2, True)
#                 ]
            
#         sequence += [
#             nn.ZeroPad2d(1, 0, 1, 0),
#             nn.Conv2d(out_channels * 8, 1, 4, padding=1, bias=False)
#             ]
            
#         self.model = nn.Sequential(*sequence)
        
#     def forward(self, x):
#         return self.model(x)
        
    
class Discriminator(nn.Module):
    def __init__(self, opt, in_channels=3, out_channels=64):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))   # InstanceNorm
                #layers.append(nn.BatchNorm2d(out_filters))     # BatchNorm
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, out_channels, normalization=False),
            *discriminator_block(out_channels, out_channels * 2),
            *discriminator_block(out_channels * 2, out_channels * 4),
            *discriminator_block(out_channels * 4, out_channels * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(out_channels * 8, 1, 4, padding=1, bias=False)
        )

    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        return self.model(x)
    
class GANLoss(nn.Module):
    def __init__(self, opt, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = opt.gan_mode
        
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == 'vanila':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode in ['wganpg']:
            self.loss = None
        else:
            raise NotImplementedError()
            
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
            
        return target_tensor.expand_as(prediction)

            
    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanila']:
            target_tensor = self.get_target_tensor(prediction, target_is_real).cuda()
            loss = self.loss(prediction, target_tensor)
        else:
            pass
        
        return loss
    
    
class TOPLoss(nn.Module):
    def __init__(self, opt):
        super(TOPLoss, self).__init__()
        
        self.vgg = VGG16(opt.top_n_layer).cuda()
        
        self.top_n_layer = opt.top_n_layer 
        self.loss = nn.MSELoss()
        
    def forward(self, real_C, rec_C):
        real_out = self.vgg(real_C)
        rec_out  = self.vgg(rec_C)
        
        losses = 0
        
        feature_list = ['relu%s' % n for n in range(1, self.top_n_layer+1)]
        for feature in feature_list:
            real_feature = real_out[feature]
            rec_feature  = rec_out[feature]
            
            _, c_n, h, w = real_feature.shape
            
            loss = self.loss(real_feature, rec_feature)
            #print("loss item :", loss/(c_n*h*w))
            losses += loss/(c_n*h*w)
                    
        return losses
        
    
def init_weights(net, init_type='normal', init_gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, gpu_ids=[]):
    if len(gpu_ids) > 1:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
        
    init_weights(net)
    return net

def define_G(opt):
    netG = Generator(opt)
    return init_net(netG, opt.gpu_ids)


def define_D(opt):
    netD = Discriminator(opt)
    return init_net(netD, opt.gpu_ids)
















        