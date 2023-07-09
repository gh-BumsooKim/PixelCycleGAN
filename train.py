from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    
    
    opt = TrainOptions().parse()
    
    data_loader = create_dataset(opt)
    
    model       = create_model(opt)
    print("\n### Model ###\nCreated Model :", model.__class__.__name__)
    
    
    print("Loaded Dataset :", data_loader.dataset.__class__.__name__)
    print("Pixel images : {%d}, Cartoon images : {%d}" %\
          (len(data_loader.dataset.pixel_img), len(data_loader.dataset.cartoon_img)))
    
    
    total_iters = 0
    
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        
        for _iter, data in enumerate(data_loader):
            
            #print(f"iter : {_iter}, data_shape : {data['pixel_img'].shape, data['cartoon_img'].shape}")
            #[8, 3, 256, 256]
            
            total_iters += 1
            
            model.set_input(data)
            model.optimize_parameters()
            
            if _iter % opt.display_freq == 0:
                model.save_output(total_iters)
            
            # TODO
            #losses = model.get_current_losses()
            
            out = "iter : {%7d}, G_loss : {%.5f}, D_loss : {%.5f}"
            print(out % (_iter, model.loss_G.cpu().item(), model.loss_D.cpu().item()))
                
            
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)
            
            
    