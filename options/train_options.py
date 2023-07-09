import argparse
import os
import torch

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Training
        parser.add_argument('--lr',             type=float, default=0.0002, help='param')
        parser.add_argument('--beta1',          type=float, default=0.5, help='param')
        parser.add_argument('--n_iter',         type=int, default=100, help='param')
        
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--n_epochs', type=int, default=100, help='.')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='.')
        parser.add_argument('--display_freq',   type=int, default=10,  help='param')
        parser.add_argument('--save_epoch_freq',type=int, default=1,   help='param')
        parser.add_argument('--save_dir',       type=str, default='./checkpoint', help='param')
        parser.add_argument('--save_output',    type=str, default='./middle_save',help='param')
        parser.add_argument('--gpu_ids',        type=str, default='0', help='param')
        
        # Losses
        parser.add_argument('--wandb',      type=str,                   help='param')
        parser.add_argument('--lambda_sc',  type=float, default=10.0,   help='param')
        parser.add_argument('--alpha',      type=float, default=0.5,    help='param')
        parser.add_argument('--mu',         type=float, default=0.1,    help='To be experimented')
        parser.add_argument('--top_n_layer',type=int,   default=4,      help='To be experimented')
        
        
        # Generator
        parser.add_argument('--gan_mode', type=str, default='lsgan')
        parser.add_argument('--n_down', type=int, default=4, help='param without init layer')
        parser.add_argument('--n_up1',  type=int, default=4, help='param')
        parser.add_argument('--n_up2',  type=int, default=3, help='param')
        parser.add_argument('--n_up3',  type=int, default=2, help='param')
        
        # Discriminator
        parser.add_argument('--n_layers',  type=int, default=2, help='param')
        parser.add_argument('--discriminator_norm',  type=str, default="batch_norm", help='param')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        
        self.isTrain = True
        return parser