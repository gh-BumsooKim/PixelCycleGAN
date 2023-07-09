from models.pixel_cycle_gan import PixelCycleGAN


def create_model(opt):
    model = PixelCycleGAN(opt)
    print("model [%s] was created" % type(model).__name__)
    return model