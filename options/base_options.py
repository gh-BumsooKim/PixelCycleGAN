import argparse

class BaseOptions():
    def __init__(self):
        self.initialzed = False
        
        
    def initialize(self, parser):
        
        # Base
        parser.add_argument('--name',       type=str, default='test', help='param')
        parser.add_argument('--load_size',  type=int, default=256, help='param')
        
        # Dataset
        parser.add_argument('--path_pixel_dataset',     type=str, default='utils/pixel art image/*.*', help='')
        parser.add_argument('--path_cartoon_dataset',   type=str, default='utils/cartoon_all/*.*', help='')
        parser.add_argument('--batch_size',             type=int, default=1, help='')
        parser.add_argument('--batch_shuffle',          type=bool,default=True, help='')
        parser.add_argument('--num_workers',            type=int, default=4, help='')
        
        self.initialized = True
        
        return parser
    
    def gather_options(self):
        
        if not self.initialzed:
            
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            
        return parser.parse_args()
        
    
    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        
        self.opt = opt
        return self.opt