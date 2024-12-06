from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import einops
from .encoder_denoise import PlainConvEncoder
from .decoder_denoise import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

import math 

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(torch.expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = einops.rearrange(x, 'b -> b 1')
        freqs = x * einops.rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class DenoiseMLP(nn.Module):
    def __init__(self, inchans, outchans, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.time_embed_dim = 1024
        self.mlp = nn.Conv3d(inchans, outchans, 1, 1, 0)
        self.time_mlp = nn.Sequential( # [2, 1024]
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, outchans*2) # [2, 512]
            )
        
    def forward(self, hidden_state, time):
        hidden_state = self.mlp(hidden_state)
        time_embed = self.time_mlp(time)  
        time_embed = einops.rearrange(time_embed, 'b c -> b c 1 1 1') 
        scale1, shift1 = time_embed.chunk(2, dim=1)  
        
        hidden_state = hidden_state * (scale1+1) + shift1
        return hidden_state

class DenoiseMLPMultiLayer(nn.Module):
    def __init__(self, inchans, outchans, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers = []
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                layers.append(DenoiseMLP(inchans, outchans))
            else :
                layers.append(DenoiseMLP(outchans, outchans))
                
        self.layers = nn.ModuleList(layers)
        
    def forward(self, hidden_state, time):
        for i in range(self.num_layers):
            hidden_state = self.layers[i](hidden_state, time)
        
        return hidden_state
    
class DecodeHead(nn.Module):
    def __init__(self, features_per_stage, num_classes=1, n_stages=6) -> None:
        super().__init__()

        mlps = []
        time_mlps = []
        # self.time_embed_dim = 1024
        decoder_size = 32
        self.n_stages = n_stages
        for i in range(n_stages):
            
            mlps.append(DenoiseMLPMultiLayer(features_per_stage[i], decoder_size, num_layers=6))
            
        self.linear_c = nn.ModuleList(mlps)
    
        self.linear_fuse = nn.Conv3d(
            in_channels=decoder_size * (n_stages - 1),
            out_channels=decoder_size,
            kernel_size=1,
            bias=False,
        )

        self.instance_norm = nn.InstanceNorm3d(decoder_size)
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv3d(
            in_channels=decoder_size,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, hidden_states, time):

        all_hidden_states = ()
        for hidden_state, mlp in zip(hidden_states[:self.n_stages-1], self.linear_c):
           
            hidden_state = mlp(hidden_state, time)

            hidden_state = nn.functional.interpolate(
                hidden_state, size=hidden_states[0].size()[2:], mode="trilinear", align_corners=False
            )
            all_hidden_states += (hidden_state,)

        
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.instance_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        logits = self.classifier(hidden_states)

        return logits


def compute_uncer(pred_out):

    uncer_out = torch.softmax(pred_out, dim=1)
    uncer_out = torch.sum(-uncer_out * torch.log(uncer_out), dim=1)
   
    return uncer_out

class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 ddim_step=5,
                 rand_step=1,
                 bta=True,
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels+num_classes, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first,bta=bta,)
        
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, 
                                   deep_supervision=deep_supervision,
                                   nonlin_first=nonlin_first)

        self.log_snr = alpha_cosine_log_snr
        self.num_classes = num_classes
        self.timesteps=ddim_step
        self.embedding_table = nn.Embedding(num_classes, num_classes)  

        self.randsteps=rand_step
        self.time_difference=1
        self.learned_sinusoidal_dim=16
        self.sample_range=(0, 0.999)
        self.noise_schedule='cosine'
        self.diffusion="ddim"
        self.accumulation=True
        self.bit_scale = 0.001 
        # time embeddings
        time_dim = 32 
        sinu_pos_emb = LearnedSinusoidalPosEmb(self.learned_sinusoidal_dim)
        fourier_dim = self.learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

        self.index = 0
        self.uncer_step = 3 
    
    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = einops.repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    @torch.no_grad()
    def ddim_sample(self, image, embeddings=None):
        device = image.device
        b, _, d, h, w = image.shape
        time_pairs = self._get_sampling_timesteps(self.randsteps*b, device=device)
    
        image = einops.repeat(image, 'b c d h w -> (r b) c d h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps*b, self.num_classes, d, h, w), device=device)
        
        outs = list()
        self.index += 1
        index = self.index
        for idx, (times_now, times_next) in enumerate(time_pairs):
            
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)

            inputs = torch.cat([image, mask_t], dim=1)
            skips = self.encoder(inputs, input_times, embeddings=embeddings)
            pred = self.decoder(skips, input_times)
            
            mask_pred = torch.argmax(pred, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 4, 1, 2, 3)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next
            
            if self.accumulation:
                outs.append(pred[None])
        
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        
        feature_size = mask_logit.shape[2]
        mask_logit = mask_logit.reshape(shape=(self.timesteps, self.randsteps, b, feature_size, d, h, w))
        mask_logit = mask_logit.mean(dim=1)
        mask_logit = mask_logit.mean(dim=0)
        return mask_logit

    def forward(self, image, gt=None, embeddings=None, ddim=False):

        if ddim:
            return self.ddim_sample(image, embeddings=embeddings)
        
        device = image.device

        ## first forward:
        times = torch.zeros((image.shape[0],), device=device).float().uniform_(self.sample_range[0],
                                                                  self.sample_range[1])  # [bs]
        # random noise
        gt_first = gt.long()
        gt_first = self.embedding_table(gt_first)
        gt_first = gt_first.squeeze(dim=1).permute(0, 4, 1, 2, 3)
        gt_first = (torch.sigmoid(gt_first) * 2 - 1) * self.bit_scale

        noise = torch.randn_like(gt_first)

        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(image, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_first + sigma * noise

        image_first = torch.cat([image, noised_gt], dim=1)

        times_first = self.time_mlp(times)

        skips = self.encoder(image_first, times_first, embeddings=embeddings)
        
        pred_first = self.decoder(skips, times_first)

        uncer_step = self.uncer_step
        with torch.no_grad():
            times = torch.zeros((image.shape[0]*uncer_step,), device=device).float().uniform_(self.sample_range[0],
                                                                    self.sample_range[1])  # [bs]
            # random noise
            gt = gt.long()
            gt = self.embedding_table(gt)
            gt = gt.squeeze(dim=1).permute(0, 4, 1, 2, 3)
            gt = (torch.sigmoid(gt) * 2 - 1) * self.bit_scale

            gt = gt.repeat(uncer_step, 1, 1, 1, 1)
            noise = torch.randn_like(gt)

            noise_level = self.log_snr(times)
            padded_noise_level = self.right_pad_dims_to(image, noise_level)
            alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
            noised_gt = alpha * gt + sigma * noise

            image = image.repeat(uncer_step, 1, 1, 1, 1)

            image = torch.cat([image, noised_gt], dim=1)

            times = self.time_mlp(times)

            skips = self.encoder(image, times, embeddings=embeddings)
            
            pred = self.decoder(skips, times)
            
            bu, c, d, h, w = pred.shape
            pred = pred.reshape((-1, uncer_step, c, d, h, w))
            pred_mean = pred.mean(dim=1)
            uncertainty = self.compute_uncer(pred_mean)


        return pred_first, uncertainty

    def compute_uncer(self, pred_out):

        uncer_out = torch.softmax(pred_out, dim=1).clamp(0.0001, 0.9999)
        ## 计算学习比重
        uncer_out = torch.sum(-uncer_out * torch.log(uncer_out), dim=1)

        return uncer_out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

import torch.nn as nn 

unet_config = {"UNet_base_num_features": 32,
                "n_conv_per_stage_encoder": [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2
                ],
                "n_conv_per_stage_decoder": [
                    2,
                    2,
                    2,
                    2,
                    2
                ],
                "num_pool_per_axis": [
                    5,
                    5,
                    5
                ],
                "pool_op_kernel_sizes": [
                    [
                        1,
                        1,
                        1
                    ],
                    [
                        2,
                        2,
                        2
                    ],
                    [
                        2,
                        2,
                        2
                    ],
                    [
                        2,
                        2,
                        2
                    ],
                    [
                        2,
                        2,
                        2
                    ],
                    [
                        2,
                        2,
                        2
                    ]
                ],
                "conv_kernel_sizes": [
                    [
                        3,
                        3,
                        3
                    ],
                    [
                        3,
                        3,
                        3
                    ],
                    [
                        3,
                        3,
                        3
                    ],
                    [
                        3,
                        3,
                        3
                    ],
                    [
                        3,
                        3,
                        3
                    ],
                    [
                        3,
                        3,
                        3
                    ]
                ],
                "unet_max_num_features": 320,
}
num_stages = len(unet_config["conv_kernel_sizes"])

conv_or_blocks_per_stage = {
    'n_conv_per_stage': unet_config["n_conv_per_stage_encoder"],
    'n_conv_per_stage_decoder': unet_config['n_conv_per_stage_decoder']
    }
other_kwargs = {
            'conv_bias': True,
            'norm_op': nn.InstanceNorm3d,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
conv_op = nn.Conv3d

def get_nnunet3d_denoise(in_chans, out_chans, ddim_step=3, rand_step=1, bta=True):
    model = PlainConvUNet(input_channels=in_chans, 
                          n_stages=num_stages, 
                          features_per_stage=[min(unet_config["UNet_base_num_features"] * 2 ** i,
                                unet_config["unet_max_num_features"]) for i in range(num_stages)],
                          conv_op=conv_op,
                          kernel_sizes=unet_config["conv_kernel_sizes"],
                          strides=unet_config["pool_op_kernel_sizes"],
                          num_classes=out_chans,
                          ddim_step=ddim_step,
                          rand_step=rand_step,
                          bta=bta,
                          **conv_or_blocks_per_stage,
                          **other_kwargs,
                          )

    return model 