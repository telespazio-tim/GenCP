"""
pix2pix_stability_model.py
Pix2pix with cross-season geometric stability loss.

Place this in models/pix2pix_stability_model.py and invoke via --model pix2pix_stability.

Must be used with the SeasonalGeoKeyDataset class (--dataset_mode seasonal_geokey).
Each batch returns A_summer, B_summer, A_winter, B_winter for the same geographic area.

Modifications compared to the standard pix2pix_model:
- Double forward pass (summer + winter) at each step
- L1 stability loss on edges (Sobel) between fake_B_summer and fake_B_winter
  on rigid zones (urban + roads + water)
- Edge match loss with OSM on rigid zones
- Rigid mask derived by looking up OSM colors in the RGB channels of A

New command-line flags:
    --lambda_stab    weight of the stability loss (default 10.0)
    --lambda_edge    weight of the edge match (OSM) loss (default 5.0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models import networks


# Colors of "rigid" classes (urban + roads + water) in RGB 0-255.
# Must match the colors effectively present in your OSM rasters.
RIGID_RGB_COLORS = [
    # water
    (128, 204, 255),
    # urban_residential (light_purple, salt_pond)
    (238, 204, 255),
    # urban_commercial (gray, light_gray)
    (204, 204, 204),
    (230, 230, 230),
    # roads_major
    (255, 80, 80),    # red_road
    (255, 148, 77),   # orange_road
    (255, 166, 77),   # medium_orange_road
    (255, 191, 128),  # light_orange_road
    # roads_minor
    (255, 217, 179),  # tertiary_road
    (255, 243, 230),  # residential_road
    (236, 217, 198),  # foot_path
    (223, 191, 159),  # track
    (255, 230, 204),  # unclassified_road
]


class Pix2PixStabilityModel(BaseModel):
    """Pix2pix with cross-season geometric stability loss."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="seasonal_geokey")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0,
                                help="weight for L1 loss")
            parser.add_argument("--lambda_stab", type=float, default=10.0,
                                help="weight for cross-season stability loss")
            parser.add_argument("--lambda_edge", type=float, default=5.0,
                                help="weight for edge match with OSM loss")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G_GAN", "G_L1", "G_stab", "G_edge_osm", "D_real", "D_fake"]
        self.visual_names = ["real_A_summer", "fake_B_summer", "real_B_summer",
                             "real_A_winter", "fake_B_winter", "real_B_winter"]
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        # Generator (input 4 channels: RGB + season, output 3)
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
        )

        if self.isTrain:
            # Conditional discriminator: input = A (4ch) + B (3ch) = 7ch
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
            )

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

        # Precompute the rigid color tensor used for masking
        # (no register_buffer because BaseModel is not an nn.Module)
        self._rigid_colors = torch.tensor(
            RIGID_RGB_COLORS, dtype=torch.float32, device=self.device
        )  # (K, 3)

        # Precompute the Sobel kernels
        self._sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=self.device,
        ).view(1, 1, 3, 3)
        self._sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=self.device,
        ).view(1, 1, 3, 3)

    def set_input(self, input):
        """Receive a batch from SeasonalGeoKeyDataset."""
        self.real_A_summer = input["A_summer"].to(self.device)
        self.real_B_summer = input["B_summer"].to(self.device)
        self.real_A_winter = input["A_winter"].to(self.device)
        self.real_B_winter = input["B_winter"].to(self.device)
        self.image_paths = input["A_summer_paths"]  # used for saving results

    def forward(self):
        """Double forward pass: summer + winter."""
        self.fake_B_summer = self.netG(self.real_A_summer)
        self.fake_B_winter = self.netG(self.real_A_winter)

    def get_rigid_mask(self, A_rgba):
        """Derive a rigid mask (B, 1, H, W) in {0, 1} from the RGB part of A.

        A_rgba: (B, 4, H, W) where the first 3 channels are normalized RGB OSM
                in [-1, 1] and the 4th channel is the season channel.

        Strategy: for each pixel, find the closest palette color in L2 distance
        in the RGB 0-255 space. If it is a rigid color, mask=1.
        """
        # Recover RGB normalized to 0-255
        rgb = A_rgba[:, :3, :, :]  # (B, 3, H, W) in [-1, 1]
        rgb_255 = ((rgb + 1.0) * 127.5).clamp(0, 255)

        # rgb_255: (B, 3, H, W), _rigid_colors: (K, 3)
        # For each pixel, compute the L2 distance to the K rigid colors
        rgb_exp = rgb_255.unsqueeze(1)  # (B, 1, 3, H, W)
        colors_exp = self._rigid_colors.view(1, -1, 3, 1, 1)  # (1, K, 3, 1, 1)
        dist_sq = ((rgb_exp - colors_exp) ** 2).sum(dim=2)  # (B, K, H, W)
        min_dist_sq = dist_sq.min(dim=1).values  # (B, H, W)

        # Tolerance: if the distance to the closest rigid color < threshold, mark as rigid.
        # Threshold of 20 (in L2 over RGB 0-255) corresponds to ~12 per channel.
        THRESHOLD = 20.0 ** 2
        mask = (min_dist_sq < THRESHOLD).float().unsqueeze(1)  # (B, 1, H, W)
        return mask

    def sobel_edges(self, x):
        """Compute Sobel gradient magnitude over x (B, C, H, W).
        Returns (B, 1, H, W)."""
        gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        gx = F.conv2d(gray, self._sobel_x, padding=1)
        gy = F.conv2d(gray, self._sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def backward_D(self):
        """Discriminator loss across both seasons (sum)."""
        # D loss on summer
        fake_AB_s = torch.cat([self.real_A_summer, self.fake_B_summer], dim=1)
        pred_fake_s = self.netD(fake_AB_s.detach())
        real_AB_s = torch.cat([self.real_A_summer, self.real_B_summer], dim=1)
        pred_real_s = self.netD(real_AB_s)

        # D loss on winter
        fake_AB_w = torch.cat([self.real_A_winter, self.fake_B_winter], dim=1)
        pred_fake_w = self.netD(fake_AB_w.detach())
        real_AB_w = torch.cat([self.real_A_winter, self.real_B_winter], dim=1)
        pred_real_w = self.netD(real_AB_w)

        self.loss_D_fake = 0.5 * (self.criterionGAN(pred_fake_s, False)
                                + self.criterionGAN(pred_fake_w, False))
        self.loss_D_real = 0.5 * (self.criterionGAN(pred_real_s, True)
                                + self.criterionGAN(pred_real_w, True))
        loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        loss_D.backward()

    def backward_G(self):
        """Generator loss: L1 + GAN + stability + edge match (OSM)."""
        # GAN: fool D on both seasons
        fake_AB_s = torch.cat([self.real_A_summer, self.fake_B_summer], dim=1)
        fake_AB_w = torch.cat([self.real_A_winter, self.fake_B_winter], dim=1)
        pred_fake_s = self.netD(fake_AB_s)
        pred_fake_w = self.netD(fake_AB_w)
        self.loss_G_GAN = 0.5 * (self.criterionGAN(pred_fake_s, True)
                               + self.criterionGAN(pred_fake_w, True))

        # L1 on both seasons
        l1_s = self.criterionL1(self.fake_B_summer, self.real_B_summer)
        l1_w = self.criterionL1(self.fake_B_winter, self.real_B_winter)
        self.loss_G_L1 = 0.5 * (l1_s + l1_w) * self.opt.lambda_L1

        # Cross-season stability loss: edges fake_summer vs fake_winter
        # on the rigid mask (derived from summer OSM, identical to winter since
        # it is the same geographic area)
        rigid_mask = self.get_rigid_mask(self.real_A_summer)
        edges_summer = self.sobel_edges(self.fake_B_summer)
        edges_winter = self.sobel_edges(self.fake_B_winter)
        diff = (edges_summer - edges_winter).abs() * rigid_mask
        # Normalize by the mask area to avoid the loss depending on the
        # amount of rigid pixels in the batch
        mask_sum = rigid_mask.sum().clamp(min=1.0)
        self.loss_G_stab = diff.sum() / mask_sum * self.opt.lambda_stab

        # Edge match with OSM: force generated edges to match the OSM edges
        # OSM = real_A[:, :3] (RGB only, without season channel)
        osm_edges = self.sobel_edges(self.real_A_summer[:, :3])
        edge_diff = (
            ((edges_summer - osm_edges).abs() + (edges_winter - osm_edges).abs())
            * rigid_mask
        )
        self.loss_G_edge_osm = edge_diff.sum() / mask_sum * self.opt.lambda_edge

        # Total
        loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_stab + self.loss_G_edge_osm
        loss_G.backward()

    def optimize_parameters(self):
        # Double forward
        self.forward()

        # Update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()