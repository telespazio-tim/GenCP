"""
Modified pix2pixHD training script.

Coordinate-conditioned pix2pixHD training with global or local generators.
Geographic XYZ metadata are supplied separately from RGB images and added at the generator bottleneck.
Includes Windows training support and per-epoch coordinate-conditioning statistics.
"""


import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
# import fractions revision
# def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

import math # revision

def lcm(a, b):  # revision
    return abs(a * b) // math.gcd(a, b) if a and b else 0 # revision

from multiprocessing import freeze_support  # revision
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer


def main(): #revision
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    visualizer = Visualizer(opt)

    # revision: Save one interpretable geographic-conditioning summary per epoch.
    coord_log_path = os.path.join(opt.checkpoints_dir, opt.name, 'coord_conditioning_log.csv')
    os.makedirs(os.path.dirname(coord_log_path), exist_ok=True)
    if not opt.continue_train or not os.path.exists(coord_log_path):
        with open(coord_log_path, 'w') as f:
            f.write(
                'epoch,weight_norm,weight_rms,mean_coord_rms,mean_bottleneck_rms,'
                'mean_coord_ratio,mean_coord_ratio_percent\n'
            )
    if opt.fp16:    
        from apex import amp
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = core_model.optimizer_G, core_model.optimizer_D

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        # revision: Accumulate activation-based conditioning statistics across all batches in this epoch.
        coord_epoch_sum = {
            'coord_rms': 0.0,
            'bottleneck_rms': 0.0,
            'coord_ratio': 0.0
        }
        coord_epoch_count = 0
        last_coord_stats = None
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            inputs = [data['label'], data['inst'], data['image'], data['feat'], data['coord']]
            losses, generated = model(*inputs, infer=save_fake)

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(core_model.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
            else:
                loss_G.backward()          
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
            else:
                loss_D.backward()        
            optimizer_D.step()

            # revision: Measure how strongly the learned XYZ embedding contributes relative to the image bottleneck.
            if hasattr(core_model.netG, 'get_coord_stats'):
                last_coord_stats = core_model.netG.get_coord_stats()
                coord_epoch_sum['coord_rms'] += last_coord_stats['coord_rms']
                coord_epoch_sum['bottleneck_rms'] += last_coord_stats['bottleneck_rms']
                coord_epoch_sum['coord_ratio'] += last_coord_stats['coord_ratio']
                coord_epoch_count += 1

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

                # revision: A ratio of 0.10 means the XYZ embedding RMS is 10 percent of the image bottleneck RMS.
                if last_coord_stats is not None:
                    print(
                        'Coord conditioning: weight_rms=%.6e, coord_rms=%.6e, '
                        'bottleneck_rms=%.6e, ratio=%.6f (%.2f%%)' % (
                            last_coord_stats['weight_rms'],
                            last_coord_stats['coord_rms'],
                            last_coord_stats['bottleneck_rms'],
                            last_coord_stats['coord_ratio'],
                            last_coord_stats['coord_ratio_percent']
                        )
                    )
                #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                    ('synthesized_image', util.tensor2im(generated.data[0])),
                                    ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                core_model.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
        
        # end of epoch 
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # revision: Report and store epoch-level conditioning statistics for later interpretation and plotting.
        if coord_epoch_count > 0 and hasattr(core_model.netG, 'get_coord_stats'):
            final_coord_stats = core_model.netG.get_coord_stats()
            mean_coord_rms = coord_epoch_sum['coord_rms'] / coord_epoch_count
            mean_bottleneck_rms = coord_epoch_sum['bottleneck_rms'] / coord_epoch_count
            mean_coord_ratio = coord_epoch_sum['coord_ratio'] / coord_epoch_count
            mean_coord_ratio_percent = mean_coord_ratio * 100.0

            print(
                'Epoch %d coord summary: weight_norm=%.6e, weight_rms=%.6e, '
                'mean_coord_rms=%.6e, mean_bottleneck_rms=%.6e, ratio=%.6f (%.2f%%)' % (
                    epoch,
                    final_coord_stats['weight_norm'],
                    final_coord_stats['weight_rms'],
                    mean_coord_rms,
                    mean_bottleneck_rms,
                    mean_coord_ratio,
                    mean_coord_ratio_percent
                )
            )

            with open(coord_log_path, 'a') as f:
                f.write(
                    '%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n' % (
                        epoch,
                        final_coord_stats['weight_norm'],
                        final_coord_stats['weight_rms'],
                        mean_coord_rms,
                        mean_bottleneck_rms,
                        mean_coord_ratio,
                        mean_coord_ratio_percent
                    )
                )

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            core_model.save('latest')
            core_model.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            core_model.update_fixed_params()
            optimizer_G = core_model.optimizer_G

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            core_model.update_learning_rate()

if __name__ == "__main__":  # revision
    freeze_support()  # revision
    main()  # revision