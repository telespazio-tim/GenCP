import torch

def create_model(opt):
    if opt.model != 'pix2pixHD':
        raise ValueError('Only coordinate-conditioned pix2pixHD is supported.')

    from .pix2pixHD_model_bottleneck import Pix2PixHDModel, InferenceModel
    model = Pix2PixHDModel() if opt.isTrain else InferenceModel()
    model.initialize(opt)

    if opt.verbose:
        print("model [%s] was created" % model.name())

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model