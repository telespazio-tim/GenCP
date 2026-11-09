"""
Modified pix2pixHD testing script.

Coordinate-conditioned pix2pixHD inference with global or local generators.
Geographic XYZ metadata are supplied separately from RGB images.
Inference uses batch size 1, evaluation mode, and torch.no_grad().
Generated images and input visualizations are saved in the standard pix2pixHD HTML structure.

"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from collections import OrderedDict
from multiprocessing import freeze_support
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import util, html
from util.visualizer import Visualizer


def main():
    opt = TestOptions().parse(save=False)
    if opt.engine or opt.onnx or opt.export_onnx:
        raise ValueError('This test runner supports native PyTorch only; ONNX/TRT need a separate validated exporter.')
    if opt.data_type not in (16, 32):
        raise ValueError('Use --data_type 32 or 16; casting a network to uint8 is not quantization.')
    if opt.batchSize != 1:
        raise ValueError('This test runner saves one image at a time. Set --batchSize 1.')

    dataset = CreateDataLoader(opt).load_data()
    model = create_model(opt)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
    if opt.data_type == 16:
        if not opt.gpu_ids:
            raise ValueError('Use FP32 for CPU inference.')
        model.half()
    model.eval()
    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    print('coord_mode=%s; eval mode; batchSize=1' % opt.coord_mode)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break
            if opt.data_type == 16:
                data['label'] = data['label'].half()
            kwargs = {'coord': data['coord']}
            # Image is absent for ordinary test_A-only inference.
            real_image = data['image'] if opt.use_encoded_image else None
            generated = model.inference(data['label'], data['inst'], real_image, **kwargs)
            visuals = OrderedDict([
                ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                ('synthesized_image', util.tensor2im(generated[0]))
            ])
            print('process image... %s' % data['path'])
            visualizer.save_images(webpage, visuals, data['path'])
    webpage.save()
    print('Results: ' + os.path.abspath(os.path.join(web_dir, 'index.html')))


if __name__ == '__main__':
    freeze_support()
    main()
