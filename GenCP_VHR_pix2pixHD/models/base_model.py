import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:%d' % self.gpu_ids[0] if self.gpu_ids else 'cpu')
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.to(self.device)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_path = os.path.join(save_dir or self.save_dir, '%s_net_%s.pth' % (epoch_label, network_label))
        if not os.path.isfile(save_path):
            raise FileNotFoundError(save_path)
        state = torch.load(save_path, map_location='cpu', weights_only=True)
        try:
            network.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                'Checkpoint is incompatible. Check --coord_mode, --netG, --norm, '
                '--ngf, --n_downsample_global and --n_blocks_global. File: ' + save_path
            ) from exc

    def update_learning_rate():
        pass
