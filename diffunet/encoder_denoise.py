import torch 
import numpy as np
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from .block_denoise import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
import math 
import torch.nn as nn
import math 

def affinity_fusion(fea_a, fea_b):
    B, C, D, H, W = fea_b.shape

    anchor_fea = fea_a.flatten(start_dim=2)

    fea_b = fea_b.reshape((B, C, -1))
    a_sq = fea_b.pow(2).sum(1).unsqueeze(2)  # 4 576 1, where 1 is expand by unsqueeze
    ab = fea_b.transpose(1, 2) @ anchor_fea
    affinity = (2 * ab - a_sq) / math.sqrt(C)  # B, THW, HW

    # softmax operation; aligned the evaluation style
    maxes = torch.max(affinity, dim=1, keepdim=True)[0]
    x_exp = torch.exp(affinity - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    affinity = x_exp / x_exp_sum

    fea_a = torch.bmm(fea_b, affinity)
    return fea_a.view(B, C, D, H, W)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv',
                 bta=True,
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        transforms = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))

            transforms.append(nn.Conv3d(2*features_per_stage[s], features_per_stage[s], 1, 1, 0))

            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.transforms = nn.ModuleList(transforms)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.bta = bta


    def forward(self, x, temb, embeddings=None):
    
        ret = []
        index = 0
        for s in self.stages:
            x = s([x, temb])
            if embeddings is not None:
                embeddings_index = embeddings[index]
                if embeddings_index.shape[0] != x.shape[0]:
                    need_repeat = x.shape[0] // embeddings_index.shape[0]
                    embeddings_index = embeddings_index.repeat(need_repeat, 1, 1, 1, 1)

                if self.bta:
                    if index == len(self.stages) - 1:
                        x = affinity_fusion(x, embeddings_index)
                    else :
                        x = self.transforms[index](torch.cat([x, embeddings_index], dim=1))
                index += 1

            ret.append(x)

        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


