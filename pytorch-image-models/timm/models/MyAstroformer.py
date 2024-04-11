#THIS CODE IS HIGHLY ADAPTED FROM TIMM
# Importing Necessary Libraries
import numpy as np
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, replace, field
from functools import partial
from typing import Callable, Optional, Union, Tuple, List
sys.path.append("../../../")

import torch
from torch import nn
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, ConvMlp, DropPath, LayerNorm, ClassifierHead, NormMlpClassifierHead
from timm.layers import create_attn, get_act_layer, get_norm_layer, get_norm_act_layer, create_conv2d, create_pool2d
from timm.layers import trunc_normal_tf_, to_2tuple, extend_tuple, make_divisible, _assert
from timm.layers import RelPosMlp, RelPosBias, RelPosBiasTf, use_fused_attn, resize_rel_pos_bias_table 
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_function
from ._manipulate import named_apply, checkpoint_seq
from ._registry import generate_default_cfgs, register_model
from timm.models.registry import register_model

__all__ = ['MYAstroformer', 'MaxxVitConvCfg', 'MaxxVitTransformerCfg', 'MaxxVit']


@dataclass
class MaxxVitTransformerCfg:
    dim_head: int = 32
    head_first: bool = True  # head ordering in qkv channel dim
    expand_ratio: float = 4.0
    expand_first: bool = True
    shortcut_bias: bool = True
    attn_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pool_type: str = 'avg2'
    rel_pos_type: str = 'bias'
    rel_pos_dim: int = 512  # for relative position types w/ MLP
    partition_ratio: int = 32
    window_size: Optional[Tuple[int, int]] = None
    grid_size: Optional[Tuple[int, int]] = None
    no_block_attn: bool = False  # disable window block attention for maxvit (ie only grid)
    use_nchw_attn: bool = False  # for MaxViT variants (not used for CoAt), keep tensors in NCHW order
    init_values: Optional[float] = None
    act_layer: str = 'gelu'
    norm_layer: str = 'layernorm2d'
    norm_layer_cl: str = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.grid_size is not None:
            self.grid_size = to_2tuple(self.grid_size)
        if self.window_size is not None:
            self.window_size = to_2tuple(self.window_size)
            if self.grid_size is None:
                self.grid_size = self.window_size


@dataclass
class MaxxVitConvCfg:
# Configuration class for the transformer component of the MaxxVit Model

    block_type: str = 'mbconv' # Dimension of each head
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    padding: str = ''
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None

    def __post_init__(self):
        # Post-initialization to ensure `grid_size` and `window_size` are properly set
        assert self.block_type in ('mbconv', 'convnext')
        use_mbconv = self.block_type == 'mbconv'
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


@dataclass
class MYAstroformer:
    # Overall configuration for MYAtroformer model defining structure and hyperparameters

    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)  # Embedding dimensions for each stage of the model
    depths: Tuple[int, ...] = (2, 3, 5, 2)  # Number of blocks in each stage
    block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('C', 'C', 'C', 'T')  # Type of blocks used in each stage, C for Conv and T for Transformer
    stem_width: Union[int, Tuple[int, int]] = 64  # Number of channels in the stem, can be a tuple for models with two stem layers
    stem_bias: bool = False  # Whether the stem layers include a bias term
    conv_cfg: MaxxVitConvCfg = field(default_factory=MaxxVitConvCfg)  # Configuration for convolutional blocks
    transformer_cfg: MaxxVitTransformerCfg = field(default_factory=MaxxVitTransformerCfg)  # Configuration for transformer blocks
    head_hidden_size: int = None  # Size of the hidden layer in the classification head, if applicable
    weight_init: str = 'vit_eff'  # Weight initialization strategy


class Attention2d(nn.Module):
    fused_attn: Final[bool]

    """ Implements multi-head attention for 2D NCHW tensors, allowing attention to operate directly on spatial features. """
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,  # Output dimension, defaults to input dimension if not provided
            dim_head: int = 32,  # Dimension of each head in the multi-head attention mechanism
            bias: bool = True,  # Whether to include bias in QKV and projection layers
            expand_first: bool = True,  # Determines whether to expand dimension at the beginning
            head_first: bool = True,  # Orders the head dimension first in the QKV computation
            rel_pos_cls: Callable = None,  # Relative position class, for incorporating positional information
            attn_drop: float = 0.,  # Dropout rate for attention weights
            proj_drop: float = 0.  # Dropout rate after projection
    ):
        super().__init__()
        dim_out = dim_out or dim  # Default output dimension to input dimension if not specified
        dim_attn = dim_out if expand_first else dim  # Determine attention dimension based on expansion flag
        self.num_heads = dim_attn // dim_head  # Calculate the number of attention heads
        self.dim_head = dim_head  # Dimension of each attention head
        self.head_first = head_first  # Head dimension ordering flag
        self.scale = dim_head ** -0.5  # Scale factor for dot-product attention
        self.fused_attn = use_fused_attn()  # Determine if fused attention implementation should be used

        # Define query, key, and value projections in a single convolution
        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias)
        # Optional relative positional embedding
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        # Attention and projection dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape  # Unpack input dimensions

        # Reshape or transpose QKV based on head ordering preference
        if self.head_first:
            q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
        else:
            q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

        # Compute attention using either fused or standard approach
        if self.fused_attn:
            attn_bias = self.rel_pos.get_bias() if self.rel_pos is not None else shared_rel_pos
            x = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2).contiguous(),
                k.transpose(-1, -2).contiguous(),
                v.transpose(-1, -2).contiguous(),
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ).transpose(-1, -2).reshape(B, -1, H, W)
        else:
            q *= self.scale  # Apply scaling
            attn = q.transpose(-2, -1) @ k
            if self.rel_pos is not None:
                attn = self.rel_pos(attn)
            elif shared_rel_pos is not None:
                attn = attn + shared_rel_pos
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)

        # Apply output projection and dropout
        x = self.proj_drop(self.proj(x))
        return x


class AttentionCl(nn.Module):
    """Implements multi-head attention where input and output are in channels-last format (B, ..., C)."""

    fused_attn: Final[bool]  # Indicates if a fused/fast attention implementation should be used

    def __init__(
            self,
            dim: int,  # Dimension of input features
            dim_out: Optional[int] = None,  # Dimension of output features, defaults to `dim` if None
            dim_head: int = 32,  # Dimension of each head in the multi-head attention
            bias: bool = True,  # Whether bias should be added to the linear layers
            expand_first: bool = True,  # Controls dimension expansion in the attention mechanism
            head_first: bool = True,  # Determines the ordering of head dimension in QKV computations
            rel_pos_cls: Callable = None,  # Class for relative positional embeddings
            attn_drop: float = 0.,  # Dropout rate for attention weights
            proj_drop: float = 0.  # Dropout rate after the projection layer
    ):
        super().__init__()
        dim_out = dim_out or dim  # Set `dim_out` to `dim` if it is not provided
        dim_attn = dim_out if expand_first and dim_out > dim else dim  # Determine the dimension for attention computation
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head  # Number of attention heads
        self.dim_head = dim_head  # Dimension of each attention head
        self.head_first = head_first  # Flag to indicate if head dimension comes first
        self.scale = dim_head ** -0.5  # Scaling factor for attention scores
        self.fused_attn = use_fused_attn()  # Check if a fused attention implementation should be used

        # Linear layers for computing queries, keys, and values
        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        # Optional relative positional embedding
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        # Dropout layers for attention and after projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B = x.shape[0]  # Batch size
        restore_shape = x.shape[:-1]  # Shape to restore output to match input (excluding channel dim)

        # Compute Q, K, V based on head dimension ordering preference
        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

        # Apply attention mechanism (fused or standard) with optional relative positional bias
        if self.fused_attn:
            attn_bias = None
            if self.rel_pos is not None:
                attn_bias = self.rel_pos.get_bias()
            elif shared_rel_pos is not None:
                attn_bias = shared_rel_pos
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale  # Scale queries
            attn = q @ k.transpose(-2, -1)  # Compute dot-product attention
            if self.rel_pos is not None:
                attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
            elif shared_rel_pos is not None:
                attn += shared_rel_pos
            attn = self.attn_drop(attn.softmax(dim=-1))  # Apply dropout after softmax
            x = attn @ v  # Weighted sum of values

        # Reshape and apply final projection
        x = x.transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj_drop(self.proj(x))  # Apply projection and dropout
        return x


class LayerScale(nn.Module):
    # LayerScale introduces a learnable scaling parameter for each channel.
    # This technique is used to stabilize and improve the training of deep networks.

    def __init__(self, dim, init_values=1e-5, inplace=False):
        """
        Initializes the LayerScale module.
        :param dim: Number of features (channels) in the input.
        :param init_values: Initial value for the scaling parameters. Small values (e.g., 1e-5) are used to start close to an identity function.
        :param inplace: If True, modifies the input tensor directly, saving memory but altering the input tensor.
        """
        super().__init__()
        self.inplace = inplace  # Flag to perform operations in-place to save memory
        self.gamma = nn.Parameter(init_values * torch.ones(dim))  # Learnable scaling parameters initialized close to 0

    def forward(self, x):
        """
        Scales the input tensor by the learned scaling parameters.
        :param x: Input tensor with shape (batch_size, num_channels, ...).
        :return: Scaled tensor.
        """
        gamma = self.gamma
        # Apply scaling; if inplace=True, modify `x` directly, otherwise return scaled tensor
        return x.mul_(gamma) if self.inplace else x * gamma


class LayerScale2d(nn.Module):
    """
    LayerScale2d module applies a learnable per-channel scaling factor to 2D convolutional layers.
    This can be used to stabilize the training of deep convolutional networks by scaling the outputs
    of convolutional layers before activation functions.

    The scaling parameters are initialized close to zero to start training with an almost identity function,
    gradually allowing the network to increase the magnitude of scaling where beneficial.
    """

    def __init__(self, dim, init_values=1e-5, inplace=False):
        """
        Initializes the LayerScale2d module.

        Args:
            dim (int): Number of channels in the input tensor to which scaling will be applied.
            init_values (float, optional): Initial value for the scaling parameters. Defaults to 1e-5.
            inplace (bool, optional): If True, modifies the input tensor in-place, reducing memory usage but potentially
                                      altering the input tensor. Defaults to False.
        """
        super().__init__()
        self.inplace = inplace  # Flag indicating whether the operation is in-place
        self.gamma = nn.Parameter(init_values * torch.ones(dim))  # Learnable scaling parameters

    def forward(self, x):
        """
        Forward pass of the LayerScale2d module.

        Scales the input tensor by the learned scaling parameters. If `inplace` is True,
        the input tensor is modified directly. Otherwise, a scaled copy of the input tensor is returned.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            Tensor: Scaled tensor with the same shape as the input.
        """
        gamma = self.gamma.view(1, -1, 1, 1)  # Reshape gamma to match the dimensions of `x` for broadcasting
        # Apply scaling; if inplace=True, modify `x` directly, otherwise return a new scaled tensor
        return x.mul_(gamma) if self.inplace else x * gamma


class Downsample2d(nn.Module):
    """
    Downsample2d module provides downsampling functionality with support for different pooling operations.
    This module can perform both spatial downsampling and channel expansion, making it useful for transitioning
    between convolutional blocks in a neural network that change the spatial dimensions and the number of channels.

    Supported pooling operations include max pooling and average pooling with different kernel sizes and strides.
    """

    def __init__(
            self,
            dim: int,  # Number of input channels
            dim_out: int,  # Number of output channels
            pool_type: str = 'avg2',  # Type of pooling operation to use
            padding: str = '',  # Padding strategy. If empty, defaults are used based on pool_type
            bias: bool = True,  # Whether to include bias in channel expansion convolution
    ):
        super().__init__()
        # Validate pool_type argument and initialize the appropriate pooling layer
        assert pool_type in ('max', 'max2', 'avg', 'avg2')
        # Select the pooling layer based on pool_type argument
        if pool_type == 'max':
            self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=padding or 1)
        elif pool_type == 'max2':
            self.pool = create_pool2d('max', 2, padding=padding or 0)  # Kernel size and stride set to 2
        elif pool_type == 'avg':
            self.pool = create_pool2d('avg', kernel_size=3, stride=2, count_include_pad=False, padding=padding or 1)
        else:  # 'avg2'
            self.pool = create_pool2d('avg', 2, padding=padding or 0)  # Kernel size and stride set to 2

        # Initialize a 1x1 convolution for channel expansion if necessary, otherwise use identity
        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        # Apply pooling to downsample spatial dimensions
        # Expand channels if necessary
        x = self.expand(self.pool(x))
        return x


def _init_transformer(module, name, scheme=''):
    """
    Initializes weights for transformer and convolution layers based on specified initialization scheme.

    Args:
        module: The module to be initialized.
        name: The name of the module.
        scheme: The name of the initialization scheme to apply.
    """
    # Initialize Conv2d and Linear layers according to the specified scheme
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if scheme == "normal":
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "trunc_normal":
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if "mlp" in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)


class TransformerBlock2d(nn.Module):
    """
    Implements a transformer block with optional 2D downsampling, designed for 2D input tensors in NCHW format.
    This block combines normalization, a multi-head self-attention mechanism, and an MLP, 
    along with residual connections around each. Optionally applies downsampling to match stride requirements.
    
    This implementation is optimized for performance on both GPUs and TPUs, 
    balancing compute efficiency with memory usage by retaining NCHW format throughout.
    """

    def __init__(
            self,
            dim: int,  # Number of input channels
            dim_out: int,  # Number of output channels
            stride: int = 1,  # Stride for downsampling. If 2, spatial downsampling is applied.
            rel_pos_cls: Callable = None,  # Class for relative positional embedding. None for absolute positioning.
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),  # Configuration object for the transformer block
            drop_path: float = 0.,  # Dropout probability for DropPath regularization
    ):
        super().__init__()
        # Define normalization and activation layers as per configuration
        norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)
        act_layer = get_act_layer(cfg.act_layer)

        # If stride is set to 2, apply downsampling for both shortcut connection and after the first normalization
        if stride == 2:
            self.shortcut = Downsample2d(dim, dim_out, pool_type=cfg.pool_type, bias=cfg.shortcut_bias)
            self.norm1 = nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('down', Downsample2d(dim, dim, pool_type=cfg.pool_type)),
            ]))
        else:
            self.shortcut = nn.Identity()  # Use identity for shortcut if no downsampling is needed
            self.norm1 = norm_layer(dim)

        # Initialize attention and MLP components with LayerScale and DropPath for regularization
        self.attn = Attention2d(
            dim, dim_out, dim_head=cfg.dim_head, expand_first=cfg.expand_first,
            bias=cfg.attn_bias, rel_pos_cls=rel_pos_cls, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop
        )
        self.ls1 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = ConvMlp(
            in_features=dim_out, hidden_features=int(dim_out * cfg.expand_ratio),
            act_layer=act_layer, drop=cfg.proj_drop)
        self.ls2 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        # Custom weight initialization function
        named_apply(partial(_init_transformer, scheme=scheme), self)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        # Forward pass applying attention and MLP blocks with residual connections
        x = self.shortcut(x) + self.drop_path1(self.ls1(self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def _init_conv(module, name, scheme=''):
    # Initialize convolution layers based on specified scheme
    # Different initialization strategies are supported, including normal, truncated normal, and xavier normal.
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def num_groups(group_size, channels):
    # Determines the number of groups for group convolution based on the group size and number of channels.
    # Special case: group_size == 1 implies depthwise convolution.
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class MbConvBlock(nn.Module):
    """
    Implements an inverted bottleneck convolution block with a pre-norm structure.
    This block is characterized by a sequence of 1x1 convolution (expansion), kxk depthwise convolution,
    and another 1x1 convolution (projection), optionally with squeeze-and-excitation attention.

    The block supports strides for downsampling and is configurable for different types of attention mechanisms.
    This makes it a versatile component for mobile networks, allowing for efficient computation and capacity scaling.
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: float = 0.
    ):
        super(MbConvBlock, self).__init__()
        norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
        mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio)
        groups = num_groups(cfg.group_size, mid_chs)

        if stride == 2:
            self.shortcut = Downsample2d(
                in_chs, out_chs, pool_type=cfg.pool_type, bias=cfg.output_bias, padding=cfg.padding)
        else:
            self.shortcut = nn.Identity()

        assert cfg.stride_mode in ('pool', '1x1', 'dw')
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if cfg.stride_mode == 'pool':
            # NOTE this is not described in paper, experiment to find faster option that doesn't stride in 1x1
            stride_pool, dilation_2 = stride, dilation[1]
            # FIXME handle dilation of avg pool
        elif cfg.stride_mode == '1x1':
            # NOTE I don't like this option described in paper, 1x1 w/ stride throws info away
            stride_1, dilation_2 = stride, dilation[1]
        else:
            stride_2, dilation_2 = stride, dilation[0]

        self.pre_norm = norm_act_layer(in_chs, apply_act=cfg.pre_norm_act)
        if stride_pool > 1:
            self.down = Downsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type, padding=cfg.padding)
        else:
            self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=stride_1)
        self.norm1 = norm_act_layer(mid_chs)

        self.conv2_kxk = create_conv2d(
            mid_chs, mid_chs, cfg.kernel_size,
            stride=stride_2, dilation=dilation_2, groups=groups, padding=cfg.padding)

        attn_kwargs = {}
        if isinstance(cfg.attn_layer, str):
            if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
                attn_kwargs['act_layer'] = cfg.attn_act_layer
                attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

        # two different orderings for SE and norm2 (due to some weights and trials using SE before norm2)
        if cfg.attn_early:
            self.se_early = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)
            self.norm2 = norm_act_layer(mid_chs)
            self.se = None
        else:
            self.se_early = None
            self.norm2 = norm_act_layer(mid_chs)
            self.se = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)

        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=cfg.output_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        shortcut = self.shortcut(x)
    
        x = self.down(self.pre_norm(x))

        # 1x1 expansion conv & norm-act
        x = self.norm1(self.conv1_1x1(x))

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.norm2(x)
        if self.se is not None:
            x = self.se(x)

        # 1x1 linear projection to output width
        x = self.drop_path(self.conv3_1x1(x)) + shortcut
        return x


class ConvNeXtBlock(nn.Module):
    """
    Implements the ConvNeXt Block, which is characterized by a large-kernel depthwise convolution
    followed by LayerNorm and an MLP. This structure is inspired by the ConvNeXt architecture,
    which utilizes simplified block design for efficient computation.

    Parameters:
    - in_chs (int): Number of input channels.
    - out_chs (Optional[int]): Number of output channels, defaults to in_chs if not provided.
    - kernel_size (int): Size of the kernel for the depthwise convolution.
    - stride (int): Stride for the depthwise convolution, used for downsampling.
    - dilation (Tuple[int, int]): Dilation for convolution operations.
    - cfg (MaxxVitConvCfg): Configuration object containing block settings.
    - conv_mlp (bool): Flag to choose between ConvMlp and Mlp for the MLP layer.
    - drop_path (float): Stochastic depth rate for regularization.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 7,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            conv_mlp: bool = True,
            drop_path: float = 0.
    ):
        super().__init__()
        out_chs = out_chs or in_chs  # Default output channels to input channels if not specified
        act_layer = get_act_layer(cfg.act_layer)
        # Select between ConvMlp and Mlp based on conv_mlp flag
        if conv_mlp:
            norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)
            mlp_layer = ConvMlp
        else:
            assert 'layernorm' in cfg.norm_layer
            norm_layer = LayerNorm
            mlp_layer = Mlp
        self.use_conv_mlp = conv_mlp  # Store whether ConvMlp or Mlp is used for later reference

        # Configure shortcut connection for identity mapping or downsampling
        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs)  # Downsample if stride is 2
        elif in_chs != out_chs:
            self.shortcut = nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=cfg.output_bias)  # Change channel dimension if necessary
        else:
            self.shortcut = nn.Identity()  # Use identity mapping if channel dimensions are the same and stride is 1

        # Determine how to apply stride and dilation based on the stride_mode
        if cfg.stride_mode == 'pool':
            stride_pool = stride  # Use pooling for downsampling
        else:
            stride_dw = stride  # Use depthwise convolution for downsampling

        # Configuring the downsample operation if necessary
        if stride_pool == 2:
            self.down = Downsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type)  # Pooling based downsampling
        else:
            self.down = nn.Identity()  # No downsampling

        # Depthwise convolution with large kernel size
        self.conv_dw = create_conv2d(
            in_chs, out_chs, kernel_size=kernel_size, stride=stride_dw, dilation=dilation[1],
            depthwise=True, bias=cfg.output_bias)
        self.norm = norm_layer(out_chs)  # Normalization layer
        # MLP layer, could be either ConvMlp or Mlp based on the conv_mlp flag
        self.mlp = mlp_layer(out_chs, int(cfg.expand_ratio * out_chs), bias=cfg.output_bias, act_layer=act_layer)
        # LayerScale (or LayerScale2d) is used for conditional scaling of the block's output
        if conv_mlp:
            self.ls = LayerScale2d(out_chs, cfg.init_values) if cfg.init_values else nn.Identity()
        else:
            self.ls = LayerScale(out_chs, cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # Optional drop path for regularization

    def forward(self, x):
        """
        Forward pass of the ConvNeXt Block.

        Parameters:
        - x: Input tensor of shape (B, C, H, W).

        Returns:
        - Output tensor after processing through the ConvNeXt block.
        """
        shortcut = self.shortcut(x)  # Apply shortcut connection
        x = self.down(x)  # Optionally downsample
        x = self.conv_dw(x)  # Apply depthwise convolution
        if self.use_conv_mlp:
            # For ConvMlp, apply normalization, mlp, and layer scale in the standard order
            x = self.norm(x)
            x = self.mlp(x)
            x = self.ls(x)
        else:
            # For Mlp, permute dimensions for layer normalization, apply mlp and layer scale, then permute back
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = self.ls(x)
            x = x.permute(0, 3, 1, 2)

        x = self.drop_path(x) + shortcut  # Apply drop path and add the shortcut
        return x


def window_partition(x, window_size: List[int]):
    """
    Partitions the input tensor into windows of a specified size.

    Parameters:
    - x: Input tensor of shape (B, H, W, C), where B is batch size, H is height, W is width, and C is channel count.
    - window_size: List specifying the height and width of the windows.

    Returns:
    - A tensor where each window is viewed as a separate element, reshaped to (-1, window_size[0], window_size[1], C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)  # Partition into windows
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)  # Reshape for separate processing
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: List[int], img_size: List[int]):
    """
    Reverses the arrangement of image windows back into the original image structure.

    Args:
        windows (Tensor): Tensor containing the image windows.
        window_size (List[int]): Dimensions of a single window (height, width).
        img_size (List[int]): Dimensions of the overall image (height, width).

    Returns:
        Tensor: The reconstructed image with the window arrangement reversed.
    """
    H, W = img_size
    C = windows.shape[-1]# Get the number of channels in the image
    # Reshape the windows tensor to match the original image layout
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
     # Rearrange the windows spatially to their original positions
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: List[int]):
    """
    Divides an image into a grid of non-overlapping windows (patches).

    Args:
        x (Tensor): The input image tensor.
        grid_size (List[int]): Dimensions of the grid (number of rows, number of columns).

    Returns:
        Tensor: A tensor containing the extracted image windows.
    """
    B, H, W, C = x.shape
    # Reshape the image to reflect the grid structure
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
     # Arrange the windows for convenient processing
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    """
    Reconstructs an image from a sequence of windows (patches).

    Args:
        windows (Tensor): A tensor containing the image windows.
        grid_size (List[int]):  A list defining the dimensions of the grid (number of rows, number of columns). 
        img_size (List[int]): A list defining the overall image dimensions (height, width). 

    Returns:
        Tensor: A tensor representing the reconstructed image.
    """
    H, W = img_size
    C = windows.shape[-1] # Get the number of channels
     # Reshape the windows back into a grid structure
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    # Rearrange the grid elements into the correct image layout
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


def get_rel_pos_cls(cfg: MaxxVitTransformerCfg, window_size):
    """
    Selects an appropriate relative position encoding class based on configuration.

    Args:
        cfg (MaxxVitTransformerCfg): Configuration object.
        window_size (List[int]): Size of the windows used in the model.

    Returns:
        partial: A partial function object, ready to create an instance of the selected 
                 relative position encoding class.
    """

    rel_pos_cls = None
    if cfg.rel_pos_type == 'mlp':
        # Select the RelPosMlp class for MLP-based relative position encoding
        rel_pos_cls = partial(RelPosMlp, window_size=window_size, hidden_dim=cfg.rel_pos_dim)

    elif cfg.rel_pos_type == 'bias':
        # Select RelPosBias for learnable bias-based relative position encoding
        rel_pos_cls = partial(RelPosBias, window_size=window_size)
    elif cfg.rel_pos_type == 'bias_tf':
        # Select RelPosBiasTf for the Transformer-based encoding variant
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)
    return rel_pos_cls


class PartitionAttentionCl(nn.Module):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        # Get normalization layer constructor
        norm_layer = partial(get_norm_layer(cfg.norm_layer_cl), eps=cfg.norm_eps)  # NOTE this block is channels-last
        # Get activation layer constructor
        act_layer = get_act_layer(cfg.act_layer)

        self.partition_block = partition_type == 'block' # Store partitioning type choice
        # Determine partition size
        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
        # Get relative position encoding class constructor
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        # Input normalization layer
        self.norm1 = norm_layer(dim)
        # Attention module 
        self.attn = AttentionCl(
            dim,
            dim,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        # LayerScale for initialization
        self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        # DropPath for regularization
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Output normalization layer
        self.norm2 = norm_layer(dim)
        # Multilayer Perceptron (FFN)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        # Image dimensions
        img_size = x.shape[1:3]

        if self.partition_block:
            # Block-based partitioning
            partitioned = window_partition(x, self.partition_size)
        else:# Grid-style partitioning
            partitioned = grid_partition(x, self.partition_size)
        # Apply attention to the partitioned image
        partitioned = self.attn(partitioned)

        if self.partition_block:
            # Reverse block partitioning to reconstruct image
            x = window_reverse(partitioned, self.partition_size, img_size)
        else:
            # Reverse grid partitioning to reconstruct image
            x = grid_reverse(partitioned, self.partition_size, img_size)
        return x

    def forward(self, x):
        # First residual block: (partition + attention) + input
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        # Second residual block: MLP + input
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ParallelPartitionAttention(nn.Module):
    """ Experimental. Grid and Block partition + single FFN
    NxC tensor layout.
    """

    def __init__(
            self,
            dim: int,
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        assert dim % 2 == 0
        # Ensure the input dimension is even
        norm_layer = partial(get_norm_layer(cfg.norm_layer_cl), eps=cfg.norm_eps) # Create normalization layers NOTE this block is channels-last
        act_layer = get_act_layer(cfg.act_layer)# Get activation layer

        assert cfg.window_size == cfg.grid_size
        self.partition_size = to_2tuple(cfg.window_size)# Store window/grid size as a tuple
        # Get relative position class for attention
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)
        # Initialize components
        self.norm1 = norm_layer(dim)# Pre-attention normalization
        self.attn_block = AttentionCl( # Attenion over partitioned blocks/windows 
            dim,
            dim // 2,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.attn_grid = AttentionCl(# Attention over partitioned grids
            dim,
            dim // 2,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()# Layer scaling
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()# Stochastic depth (for regularization)


        self.norm2 = norm_layer(dim)# Mid-block normalization
        self.mlp = Mlp(# Multilayer perceptron (FFN) block
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):# Get image height and width
        img_size = x.shape[1:3]
        # Partition into blocks
        partitioned_block = window_partition(x, self.partition_size)
        partitioned_block = self.attn_block(partitioned_block)
        x_window = window_reverse(partitioned_block, self.partition_size, img_size)
        # Partition into grids
        partitioned_grid = grid_partition(x, self.partition_size)
        partitioned_grid = self.attn_grid(partitioned_grid)
        x_grid = grid_reverse(partitioned_grid, self.partition_size, img_size)

        return torch.cat([x_window, x_grid], dim=-1)

    def forward(self, x):
        # First residual block: (partition + attention) + input
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        # Second residual block: MLP + input
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def window_partition_nchw(x, window_size: List[int]):
    """Divides the input tensor into windows (NCHW format).

    Args:
        x: Input tensor with shape (Batch, Channels, Height, Width)
        window_size: Size of the window as a list [window_height, window_width]

    Returns:
         Tensor of windows with shape 
         (num_windows, Channels, window_height, window_width)
    """
    B, C, H, W = x.shape# Get the dimensions of the input tensor
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])# Reshape into windows:
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1]) # Rearrange dimensions and flatten for attention processing:
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse_nchw(windows, window_size: List[int], img_size: List[int]):
    """Reconstructs the original tensor from windows (NCHW format).

    Args:
        windows: Tensor of windows 
                 (num_windows, Channels, window_height, window_width) 
        window_size: Size of the window [window_height, window_width]
        img_size: Original image size [Height, Width]

    Returns:
        Reconstructed tensor with shape (Batch, Channels, Height, Width)
    """
    H, W = img_size
    C = windows.shape[1]
    # Reshape windows with correct number of rows and columns based on image size
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    # Rearrange dimensions back to the image format
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def grid_partition_nchw(x, grid_size: List[int]):
    """Splits an image tensor into non-overlapping grids (NCHW format).

    Args:
        x: Input tensor in NCHW format (Batch, Channels, Height, Width)
        grid_size: A list specifying the grid dimensions [grid_height, grid_width] 

    Returns:
        Tensor of grid patches with shape: 
        (num_grids, Channels, grid_height, grid_width) 
    """
    B, C, H, W = x.shape# Extract dimensions of the input tensor
    # Reshape the tensor by dividing it into grids:
    x = x.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Rearrange the dimensions for easier processing: 
    #  - Batch stays first
    #  - Combine the grid row and column dimensions 
    #  - Channels come next
    #  - Individual grid height and width dimensions follow
    windows = x.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, C, grid_size[0], grid_size[1])
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def grid_reverse_nchw(windows, grid_size: List[int], img_size: List[int]):
    """Reconstructs the original image tensor from grid patches (NCHW format).

    Args:
        windows: Tensor of grid patches 
                 (num_grids, Channels, grid_height, grid_width)
        grid_size: Specifies the grid dimensions [grid_height, grid_width]
        img_size: Original image dimensions [Height, Width]

    Returns:
        Reconstructed tensor in NCHW format (Batch, Channels, Height, Width)
    """
    H, W = img_size# Get original image height and width    
    C = windows.shape[1]# Number of channels
    # Reshape the grid patches, putting the grid dimensions back in place:
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], C, grid_size[0], grid_size[1])
    # Rearrange the dimensions back to the original image format (NCHW):
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous().view(-1, C, H, W)
    return x


class PartitionAttention2d(nn.Module):
    """ Grid or Block partition + Attn + FFN

    '2D' NCHW tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
         # Initialize normalization and activation layers
        norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)  # NOTE this block is channels-last
        act_layer = get_act_layer(cfg.act_layer)
        # Determine partition type (block or grid)
        self.partition_block = partition_type == 'block'
        # Set partition size
        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
        # Get relative position encoding class
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)
        # Initialize components of the attention block
        self.norm1 = norm_layer(dim)
        self.attn = Attention2d(
            dim,
            dim,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Initialize components of the MLP (FFN) block:
        self.norm2 = norm_layer(dim)
        self.mlp = ConvMlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[-2:]# Get image height and width
        # Partition the input tensor
        if self.partition_block:
            partitioned = window_partition_nchw(x, self.partition_size)# Block partitioning
        else:
            partitioned = grid_partition_nchw(x, self.partition_size)# Grid partitioning

        partitioned = self.attn(partitioned)# Apply attention to partitions
        # Reverse the partitioning process:
        if self.partition_block:
            x = window_reverse_nchw(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse_nchw(partitioned, self.partition_size, img_size)
        return x

    def forward(self, x):
        # First residual block: (partition + attention) + input
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        # Second residual block: MLP + input
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# class MaxxVitBlock(nn.Module):   #NOT NEEDED in Astroformers
#     """ MaxVit conv, window partition + FFN , grid partition + FFN
#     It combines convolutional operations with window-based and grid-based attention mechanisms for processing image data.
#     """

#     def __init__(
#             self,
#             dim: int,
#             dim_out: int,
#             stride: int = 1,
#             conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
#             transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
#             drop_path: float = 0.,
#     ):
#         super().__init__()
#         self.nchw_attn = transformer_cfg.use_nchw_attn# Store attention formatting preference
#         # Select convolutional block type
#         conv_cls = ConvNeXtBlock if conv_cfg.block_type == 'convnext' else MbConvBlock
#         self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)
#         # Attention configuration
#         attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
#         partition_layer = PartitionAttention2d if self.nchw_attn else PartitionAttentionCl
#          # Initialize attention layers with appropriate formatting
#         self.attn_block = None if transformer_cfg.no_block_attn else partition_layer(**attn_kwargs)
#         self.attn_grid = partition_layer(partition_type='grid', **attn_kwargs)

#     def init_weights(self, scheme=''):
#         # Initialize weights in attention blocks and the convolutional layer.
#         if self.attn_block is not None:
#             named_apply(partial(_init_transformer, scheme=scheme), self.attn_block)
#         named_apply(partial(_init_transformer, scheme=scheme), self.attn_grid)
#         named_apply(partial(_init_conv, scheme=scheme), self.conv)

#     def forward(self, x):
#         # NCHW format
#         x = self.conv(x)# Convolutional Processing
#         # Handle attention formatting   
#         if not self.nchw_attn:  
#             x = x.permute(0, 2, 3, 1)  # to NHWC (channels-last)
#         # Apply optional block-based attention and grid-based attention
#         if self.attn_block is not None:
#             x = self.attn_block(x)
#         x = self.attn_grid(x)
#         if not self.nchw_attn:
#             x = x.permute(0, 3, 1, 2)  # back to NCHW
#         return x


# class ParallelMaxxVitBlock(nn.Module):
#     """ MaxVit block with parallel cat(window + grid), one FF
#     Experimental timm block.
#     The key idea is to perform both window-based and grid-based attention in parallel, then combine the results before a single feed-forward network (FFN).
#     """

#     def __init__(
#             self,
#             dim,
#             dim_out,
#             stride=1,
#             num_conv=2,
#             conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
#             transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
#             drop_path=0.,
#     ):
#         super().__init__()
#         # Select convolutional block type
#         conv_cls = ConvNeXtBlock if conv_cfg.block_type == 'convnext' else MbConvBlock
#         # Build a sequence of convolutional layers if `num_conv` is greater than 1
#         if num_conv > 1:
#             convs = [conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)]
#             convs += [conv_cls(dim_out, dim_out, cfg=conv_cfg, drop_path=drop_path)] * (num_conv - 1)
#             self.conv = nn.Sequential(*convs)
#         else:# Create a single convolutional layer
#             self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)
#         # Initialize the parallel attention module
#         self.attn = ParallelPartitionAttention(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)

#     def init_weights(self, scheme=''):
#         # Initialize attention and convolutional layer weights
#         named_apply(partial(_init_transformer, scheme=scheme), self.attn)
#         named_apply(partial(_init_conv, scheme=scheme), self.conv)

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.permute(0, 2, 3, 1)# Transpose to NHWC format
#         x = self.attn(x)# Apply parallel window and grid attention
#         x = x.permute(0, 3, 1, 2)# Transpose back to NCHW format
#         return x


class MaxxVitStage(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 2,
            depth: int = 4,
            feat_size: Tuple[int, int] = (14, 14),
            block_types: Union[str, Tuple[str]] = 'C',
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: Union[float, List[float]] = 0.,
    ):
        super().__init__()
        
        self.grad_checkpointing = False
        # Configure block types (can be a single type or a sequence to specify per block)
        block_types = extend_tuple(block_types, depth)
        # Build the sequence of blocks
        blocks = []
        for i, t in enumerate(block_types):
            block_stride = stride if i == 0 else 1# First block in stage may have downsampling
            assert t in ('C', 'T', 'M', 'PM')
            if t == 'C':
                conv_cls = ConvNeXtBlock if conv_cfg.block_type == 'convnext' else MbConvBlock
                blocks += [conv_cls(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    cfg=conv_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'T':
                rel_pos_cls = get_rel_pos_cls(transformer_cfg, feat_size)
                blocks += [TransformerBlock2d(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    rel_pos_cls=rel_pos_cls,
                    cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            # elif t == 'M': #NOT NEEDED as here we only deal with convoutions and transformers
            #     blocks += [MaxxVitBlock(
            #         in_chs,
            #         out_chs,
            #         stride=block_stride,
            #         conv_cfg=conv_cfg,
            #         transformer_cfg=transformer_cfg,
            #         drop_path=drop_path[i],
            #     )]
            # elif t == 'PM':
            #     blocks += [ParallelMaxxVitBlock(
            #         in_chs,
            #         out_chs,
            #         stride=block_stride,
            #         conv_cfg=conv_cfg,
            #         transformer_cfg=transformer_cfg,
            #         drop_path=drop_path[i],
            #     )]
            in_chs = out_chs# Update input channels for the next block
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)# Gradient checkpointing
        else:
            x = self.blocks(x)# Process the input through the sequence of blocks
        return x


class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):# Ensure out_chs is a tuple
            out_chs = to_2tuple(out_chs)
        # Creates a normalization and activation layer helper
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]# Store the final output channels
        self.stride = 2 # Implies downsampling by a factor of 2
        # Components of the Stem block
        self.conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
        self.norm1 = norm_act_layer(out_chs[0])
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self) # Initialize convolutional weights

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x) # Normalize and apply activation
        x = self.conv2(x)
        return x


def cfg_window_size(cfg: MaxxVitTransformerCfg, img_size: Tuple[int, int]):
    """Configures window size parameters within the MaxxVitTransformerCfg.

    Args:
        cfg: The MaxxVitTransformerCfg configuration object.
        img_size:  A tuple representing the image dimensions (height, width).

    Returns:
        The updated configuration object (cfg).
    """
    if cfg.window_size is not None:
        assert cfg.grid_size
        return cfg
     # Calculate partition size based on image size and partition ratio 
    partition_size = (img_size[0] // cfg.partition_ratio, img_size[1] // cfg.partition_ratio)
    # Update configuration with calculated values
    cfg = replace(cfg, window_size=partition_size, grid_size=partition_size)
    return cfg


def _overlay_kwargs(cfg: MYAstroformer, **kwargs):
    """Separates keyword arguments for targeted configuration updates.

    Args:
        cfg: The MYAstroformer configuration object.
        **kwargs: Keyword arguments to be used for updating the configuration.

    Returns:
        The updated configuration object (cfg).  
    """
    transformer_kwargs = {}
    conv_kwargs = {}
    base_kwargs = {}
    # Categorize keyword arguments based on prefixes
    for k, v in kwargs.items():
        if k.startswith('transformer_'):
            transformer_kwargs[k.replace('transformer_', '')] = v
        elif k.startswith('conv_'):
            conv_kwargs[k.replace('conv_', '')] = v
        else:
            base_kwargs[k] = v
    # Update configuration sections selectively
    cfg = replace(
        cfg,
        transformer_cfg=replace(cfg.transformer_cfg, **transformer_kwargs),
        conv_cfg=replace(cfg.conv_cfg, **conv_kwargs),
        **base_kwargs
    )
    return cfg


class MaxxVit(nn.Module):
    """ CoaTNet + MaxVit base model.

    Highly configurable for different block compositions, tensor layouts, pooling types.
    """

    def __init__(
            self,
            cfg: MYAstroformer,
            img_size: Union[int, Tuple[int, int]] = 224,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            **kwargs,
    ):
        super().__init__()
        # Preprocess input image siz
        img_size = to_2tuple(img_size)
        if kwargs:
            cfg = _overlay_kwargs(cfg, **kwargs)
        # Adjust transformer config based on image size
        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = cfg.embed_dim[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = [] # Storage for feature map metadata
        # Create the stem module for initial feature extraction
        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
            padding=cfg.conv_cfg.padding,
            bias=cfg.stem_bias,
            act_layer=cfg.conv_cfg.act_layer,
            norm_layer=cfg.conv_cfg.norm_layer,
            norm_eps=cfg.conv_cfg.norm_eps,
        )
        # Calculate initial stride and feature map size
        stride = self.stem.stride
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])
        # Calculate drop path rates for each stage
        num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == num_stages
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        in_chs = self.stem.out_chs
        # Build the core stages of the network
        stages = []
        for i in range(num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages += [MaxxVitStage(
                in_chs,
                out_chs,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                drop_path=dpr[i],
            )]
            stride *= stage_stride
            in_chs = out_chs
            self.feature_info += [dict(num_chs=out_chs, reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        # Create the classification head components
        final_norm_layer = partial(get_norm_layer(cfg.transformer_cfg.norm_layer), eps=cfg.transformer_cfg.norm_eps)
        self.head_hidden_size = cfg.head_hidden_size
        if self.head_hidden_size:
            self.norm = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=self.head_hidden_size,
                pool_type=global_pool,
                drop_rate=drop_rate,
                norm_layer=final_norm_layer,
            )
        else:
            # standard classifier head w/ norm, pooling, fc classifier
            self.norm = final_norm_layer(self.num_features)
            self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if cfg.weight_init:
            named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



model_cfgs = dict(
    astroformer_5=MYAstroformer(
        embed_dim=(256, 512, 1280, 2048),
        depths=(2, 12, 28, 2),
        stem_width=192,
        head_hidden_size=2048,
    )
)
# def _rw_coat_cfg(
#         stride_mode='pool',
#         pool_type='avg2',
#         conv_output_bias=False,
#         conv_attn_early=False,
#         conv_attn_act_layer='relu',
#         conv_norm_layer='',
#         transformer_shortcut_bias=True,
#         transformer_norm_layer='layernorm2d',
#         transformer_norm_layer_cl='layernorm',
#         init_values=None,
#         rel_pos_type='bias',
#         rel_pos_dim=512,
# ):
#     # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
#     # Common differences for initial timm models:
#     # - pre-norm layer in MZBConv included an activation after norm
#     # - mbconv expansion calculated from input instead of output chs
#     # - mbconv shortcut and final 1x1 conv did not have a bias
#     # - SE act layer was relu, not silu
#     # - mbconv uses silu in timm, not gelu
#     # - expansion in attention block done via output proj, not input proj
#     # Variable differences (evolved over training initial models):
#     # - avg pool with kernel_size=2 favoured downsampling (instead of maxpool for coat)
#     # - SE attention was between conv2 and norm/act
#     # - default to avg pool for mbconv downsample instead of 1x1 or dw conv
#     # - transformer block shortcut has no bias
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             stride_mode=stride_mode,
#             pool_type=pool_type,
#             pre_norm_act=True,
#             expand_output=False,
#             output_bias=conv_output_bias,
#             attn_early=conv_attn_early,
#             attn_act_layer=conv_attn_act_layer,
#             act_layer='silu',
#             norm_layer=conv_norm_layer,
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             expand_first=False,
#             shortcut_bias=transformer_shortcut_bias,
#             pool_type=pool_type,
#             init_values=init_values,
#             norm_layer=transformer_norm_layer,
#             norm_layer_cl=transformer_norm_layer_cl,
#             rel_pos_type=rel_pos_type,
#             rel_pos_dim=rel_pos_dim,
#         ),
#     )


# def _rw_max_cfg(
#         stride_mode='dw',
#         pool_type='avg2',
#         conv_output_bias=False,
#         conv_attn_ratio=1 / 16,
#         conv_norm_layer='',
#         transformer_norm_layer='layernorm2d',
#         transformer_norm_layer_cl='layernorm',
#         window_size=None,
#         dim_head=32,
#         init_values=None,
#         rel_pos_type='bias',
#         rel_pos_dim=512,
# ):
#     # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
#     # Differences of initial timm models:
#     # - mbconv expansion calculated from input instead of output chs
#     # - mbconv shortcut and final 1x1 conv did not have a bias
#     # - mbconv uses silu in timm, not gelu
#     # - expansion in attention block done via output proj, not input proj
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             stride_mode=stride_mode,
#             pool_type=pool_type,
#             expand_output=False,
#             output_bias=conv_output_bias,
#             attn_ratio=conv_attn_ratio,
#             act_layer='silu',
#             norm_layer=conv_norm_layer,
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             expand_first=False,
#             pool_type=pool_type,
#             dim_head=dim_head,
#             window_size=window_size,
#             init_values=init_values,
#             norm_layer=transformer_norm_layer,
#             norm_layer_cl=transformer_norm_layer_cl,
#             rel_pos_type=rel_pos_type,
#             rel_pos_dim=rel_pos_dim,
#         ),
#     )


# def _next_cfg(
#         stride_mode='dw',
#         pool_type='avg2',
#         conv_norm_layer='layernorm2d',
#         conv_norm_layer_cl='layernorm',
#         transformer_norm_layer='layernorm2d',
#         transformer_norm_layer_cl='layernorm',
#         window_size=None,
#         no_block_attn=False,
#         init_values=1e-6,
#         rel_pos_type='mlp',  # MLP by default for maxxvit
#         rel_pos_dim=512,
# ):
#     # For experimental models with convnext instead of mbconv
#     init_values = to_2tuple(init_values)
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             block_type='convnext',
#             stride_mode=stride_mode,
#             pool_type=pool_type,
#             expand_output=False,
#             init_values=init_values[0],
#             norm_layer=conv_norm_layer,
#             norm_layer_cl=conv_norm_layer_cl,
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             expand_first=False,
#             pool_type=pool_type,
#             window_size=window_size,
#             no_block_attn=no_block_attn,  # enabled for MaxxViT-V2
#             init_values=init_values[1],
#             norm_layer=transformer_norm_layer,
#             norm_layer_cl=transformer_norm_layer_cl,
#             rel_pos_type=rel_pos_type,
#             rel_pos_dim=rel_pos_dim,
#         ),
#     )


# def _tf_cfg():
#     return dict(
#         conv_cfg=MaxxVitConvCfg(
#             norm_eps=1e-3,
#             act_layer='gelu_tanh',
#             padding='same',
#         ),
#         transformer_cfg=MaxxVitTransformerCfg(
#             norm_eps=1e-5,
#             act_layer='gelu_tanh',
#             head_first=False,  # heads are interleaved (q_nh, q_hdim, k_nh, q_hdim, ....)
#             rel_pos_type='bias_tf',
#         ),
#     )


# model_cfgs = dict(
#     # timm specific CoAtNet configs
#     coatnet_pico_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(2, 3, 5, 2),
#         stem_width=(32, 64),
#         **_rw_max_cfg(  # using newer max defaults here
#             conv_output_bias=True,
#             conv_attn_ratio=0.25,
#         ),
#     ),
#     coatnet_nano_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(3, 4, 6, 3),
#         stem_width=(32, 64),
#         **_rw_max_cfg(  # using newer max defaults here
#             stride_mode='pool',
#             conv_output_bias=True,
#             conv_attn_ratio=0.25,
#         ),
#     ),
#     coatnet_0_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 3, 7, 2),  # deeper than paper '0' model
#         stem_width=(32, 64),
#         **_rw_coat_cfg(
#             conv_attn_early=True,
#             transformer_shortcut_bias=False,
#         ),
#     ),
#     coatnet_1_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 6, 14, 2),
#         stem_width=(32, 64),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             conv_attn_early=True,
#             transformer_shortcut_bias=False,
#         )
#     ),
#     coatnet_2_rw=MaxxVitCfg(
#         embed_dim=(128, 256, 512, 1024),
#         depths=(2, 6, 14, 2),
#         stem_width=(64, 128),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             conv_attn_act_layer='silu',
#             #init_values=1e-6,
#         ),
#     ),
#     coatnet_3_rw=MaxxVitCfg(
#         embed_dim=(192, 384, 768, 1536),
#         depths=(2, 6, 14, 2),
#         stem_width=(96, 192),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             conv_attn_act_layer='silu',
#             init_values=1e-6,
#         ),
#     ),

#     # Experimental CoAtNet configs w/ ImageNet-1k train (different norm layers, MLP rel-pos)
#     coatnet_bn_0_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 3, 7, 2),  # deeper than paper '0' model
#         stem_width=(32, 64),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             conv_attn_early=True,
#             transformer_shortcut_bias=False,
#             transformer_norm_layer='batchnorm2d',
#         )
#     ),
#     coatnet_rmlp_nano_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(3, 4, 6, 3),
#         stem_width=(32, 64),
#         **_rw_max_cfg(
#             conv_output_bias=True,
#             conv_attn_ratio=0.25,
#             rel_pos_type='mlp',
#             rel_pos_dim=384,
#         ),
#     ),
#     coatnet_rmlp_0_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 3, 7, 2),  # deeper than paper '0' model
#         stem_width=(32, 64),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             rel_pos_type='mlp',
#         ),
#     ),
#     coatnet_rmlp_1_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 6, 14, 2),
#         stem_width=(32, 64),
#         **_rw_coat_cfg(
#             pool_type='max',
#             conv_attn_early=True,
#             transformer_shortcut_bias=False,
#             rel_pos_type='mlp',
#             rel_pos_dim=384,  # was supposed to be 512, woops
#         ),
#     ),
#     coatnet_rmlp_1_rw2=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 6, 14, 2),
#         stem_width=(32, 64),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             rel_pos_type='mlp',
#             rel_pos_dim=512,  # was supposed to be 512, woops
#         ),
#     ),
#     coatnet_rmlp_2_rw=MaxxVitCfg(
#         embed_dim=(128, 256, 512, 1024),
#         depths=(2, 6, 14, 2),
#         stem_width=(64, 128),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             conv_attn_act_layer='silu',
#             init_values=1e-6,
#             rel_pos_type='mlp'
#         ),
#     ),
#     coatnet_rmlp_3_rw=MaxxVitCfg(
#         embed_dim=(192, 384, 768, 1536),
#         depths=(2, 6, 14, 2),
#         stem_width=(96, 192),
#         **_rw_coat_cfg(
#             stride_mode='dw',
#             conv_attn_act_layer='silu',
#             init_values=1e-6,
#             rel_pos_type='mlp'
#         ),
#     ),

#     coatnet_nano_cc=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(3, 4, 6, 3),
#         stem_width=(32, 64),
#         block_type=('C', 'C', ('C', 'T'), ('C', 'T')),
#         **_rw_coat_cfg(),
#     ),
#     coatnext_nano_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(3, 4, 6, 3),
#         stem_width=(32, 64),
#         weight_init='normal',
#         **_next_cfg(
#             rel_pos_type='bias',
#             init_values=(1e-5, None)
#         ),
#     ),

#     # Trying to be like the CoAtNet paper configs
#     coatnet_0=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 3, 5, 2),
#         stem_width=64,
#         head_hidden_size=768,
#     ),
#     coatnet_1=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 6, 14, 2),
#         stem_width=64,
#         head_hidden_size=768,
#     ),
#     coatnet_2=MaxxVitCfg(
#         embed_dim=(128, 256, 512, 1024),
#         depths=(2, 6, 14, 2),
#         stem_width=128,
#         head_hidden_size=1024,
#     ),
#     coatnet_3=MaxxVitCfg(
#         embed_dim=(192, 384, 768, 1536),
#         depths=(2, 6, 14, 2),
#         stem_width=192,
#         head_hidden_size=1536,
#     ),
#     coatnet_4=MaxxVitCfg(
#         embed_dim=(192, 384, 768, 1536),
#         depths=(2, 12, 28, 2),
#         stem_width=192,
#         head_hidden_size=1536,
#     ),
#     coatnet_5=MaxxVitCfg(
#         embed_dim=(256, 512, 1280, 2048),
#         depths=(2, 12, 28, 2),
#         stem_width=192,
#         head_hidden_size=2048,
#     ),

#     # Experimental MaxVit configs
#     maxvit_pico_rw=MaxxVitCfg(
#         embed_dim=(32, 64, 128, 256),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=(24, 32),
#         **_rw_max_cfg(),
#     ),
#     maxvit_nano_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(1, 2, 3, 1),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         **_rw_max_cfg(),
#     ),
#     maxvit_tiny_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         **_rw_max_cfg(),
#     ),
#     maxvit_tiny_pm=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(2, 2, 5, 2),
#         block_type=('PM',) * 4,
#         stem_width=(32, 64),
#         **_rw_max_cfg(),
#     ),

#     maxvit_rmlp_pico_rw=MaxxVitCfg(
#         embed_dim=(32, 64, 128, 256),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=(24, 32),
#         **_rw_max_cfg(rel_pos_type='mlp'),
#     ),
#     maxvit_rmlp_nano_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(1, 2, 3, 1),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         **_rw_max_cfg(rel_pos_type='mlp'),
#     ),
#     maxvit_rmlp_tiny_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         **_rw_max_cfg(rel_pos_type='mlp'),
#     ),
#     maxvit_rmlp_small_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         **_rw_max_cfg(
#             rel_pos_type='mlp',
#             init_values=1e-6,
#         ),
#     ),
#     maxvit_rmlp_base_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 6, 14, 2),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         head_hidden_size=768,
#         **_rw_max_cfg(
#             rel_pos_type='mlp',
#         ),
#     ),

#     maxxvit_rmlp_nano_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(1, 2, 3, 1),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         weight_init='normal',
#         **_next_cfg(),
#     ),
#     maxxvit_rmlp_tiny_rw=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=(32, 64),
#         **_next_cfg(),
#     ),
#     maxxvit_rmlp_small_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=(48, 96),
#         **_next_cfg(),
#     ),

#     maxxvitv2_nano_rw=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(1, 2, 3, 1),
#         block_type=('M',) * 4,
#         stem_width=(48, 96),
#         weight_init='normal',
#         **_next_cfg(
#             no_block_attn=True,
#             rel_pos_type='bias',
#         ),
#     ),
#     maxxvitv2_rmlp_base_rw=MaxxVitCfg(
#         embed_dim=(128, 256, 512, 1024),
#         depths=(2, 6, 12, 2),
#         block_type=('M',) * 4,
#         stem_width=(64, 128),
#         **_next_cfg(
#             no_block_attn=True,
#         ),
#     ),
#     maxxvitv2_rmlp_large_rw=MaxxVitCfg(
#         embed_dim=(160, 320, 640, 1280),
#         depths=(2, 6, 16, 2),
#         block_type=('M',) * 4,
#         stem_width=(80, 160),
#         head_hidden_size=1280,
#         **_next_cfg(
#             no_block_attn=True,
#         ),
#     ),

#     # Trying to be like the MaxViT paper configs
#     maxvit_tiny_tf=MaxxVitCfg(
#         embed_dim=(64, 128, 256, 512),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=64,
#         stem_bias=True,
#         head_hidden_size=512,
#         **_tf_cfg(),
#     ),
#     maxvit_small_tf=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 2, 5, 2),
#         block_type=('M',) * 4,
#         stem_width=64,
#         stem_bias=True,
#         head_hidden_size=768,
#         **_tf_cfg(),
#     ),
#     maxvit_base_tf=MaxxVitCfg(
#         embed_dim=(96, 192, 384, 768),
#         depths=(2, 6, 14, 2),
#         block_type=('M',) * 4,
#         stem_width=64,
#         stem_bias=True,
#         head_hidden_size=768,
#         **_tf_cfg(),
#     ),
#     maxvit_large_tf=MaxxVitCfg(
#         embed_dim=(128, 256, 512, 1024),
#         depths=(2, 6, 14, 2),
#         block_type=('M',) * 4,
#         stem_width=128,
#         stem_bias=True,
#         head_hidden_size=1024,
#         **_tf_cfg(),
#     ),
#     maxvit_xlarge_tf=MaxxVitCfg(
#         embed_dim=(192, 384, 768, 1536),
#         depths=(2, 6, 14, 2),
#         block_type=('M',) * 4,
#         stem_width=192,
#         stem_bias=True,
#         head_hidden_size=1536,
#         **_tf_cfg(),
#     ),
# )
#Above part not needed so Commented
# Define a function to filter and adjust model state dictionaries during checkpoint loading.
def checkpoint_filter_fn(state_dict, model: nn.Module):
    # Obtain the current model's state dictionary.
    model_state_dict = model.state_dict()
    # Initialize an empty dictionary to store the adjusted state.
    out_dict = {}
    # Iterate over the provided state dictionary.
    for k, v in state_dict.items():
        # Check if the key ends with a specific string indicating a relative position bias table.
        if k.endswith("relative_position_bias_table"):
            # Extract the corresponding submodule from the model.
            m = model.get_submodule(k[:-29])
            # Check if the shape of the provided table does not match the model's or if the window sizes are unequal.
            if (
                v.shape != m.relative_position_bias_table.shape
                or m.window_size[0] != m.window_size[1]
            ):
                # Resize the relative position bias table to match the model's expected shape and window size.
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )
        
        # Check if the key exists in the model's state dict, the dimensions differ, but the number of elements is the same.
        if (
            k in model_state_dict
            and v.ndim != model_state_dict[k].ndim
            and v.numel() == model_state_dict[k].numel()
        ):
            # Assert that the dimensions are either for a 2D (linear layer) or a 4D (conv2d layer).
            assert v.ndim in (2, 4)
            # Reshape the tensor to match the model's expected shape.
            v = v.reshape(model_state_dict[k].shape)
        # Update the output dictionary with the potentially adjusted tensor.
        out_dict[k] = v
    # Return the adjusted state dictionary.
    return out_dict

# Define a function to create a MaxxVit model variant based on provided configurations.
def _create_maxxvit(variant, cfg_variant=None, pretrained=False, **kwargs):
    # Determine the configuration variant.
    if cfg_variant is None:
        # Use the provided variant directly if it exists in the configurations, else modify it.
        if variant in model_cfgs:
            cfg_variant = variant
        else:
            cfg_variant = "_".join(variant.split("_")[:-1])
    # Build and return the model using the selected configuration.
    return build_model_with_cfg(
        MaxxVit,
        variant,
        pretrained,
        model_cfg=model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )

# Define a function to create a default model configuration.
def _cfg(url="", **kwargs):
    # Return a dictionary containing default configuration parameters for a model.
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 64, 64),
        "pool_size": (7, 7),
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "first_conv": "stem.conv1",
        "classifier": "head.fc",
        "fixed_input_size": True,
        **kwargs,
    }

# Generate default configurations for various untrained CoatNet models.
default_cfgs = generate_default_cfgs(
    { # timm specific CoAtNet configs, ImageNet-1k pretrain, fixed rel-pos
    'coatnet_pico_rw_224.untrained': _cfg(url=''),
    'coatnet_nano_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnet_nano_rw_224_sw-f53093b4.pth',
        crop_pct=0.9),
    'coatnet_0_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnet_0_rw_224_sw-a6439706.pth'),
    'coatnet_1_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnet_1_rw_224_sw-5cae1ea8.pth'
    ),

    # timm specific CoAtNet configs, ImageNet-12k pretrain w/ 1k fine-tune, fixed rel-pos
    'coatnet_2_rw_224.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    #'coatnet_3_rw_224.untrained': _cfg(url=''),

    # Experimental CoAtNet configs w/ ImageNet-12k pretrain -> 1k fine-tune (different norm layers, MLP rel-pos)
    'coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    # Experimental CoAtNet configs w/ ImageNet-1k train (different norm layers, MLP rel-pos)
    'coatnet_bn_0_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnet_bn_0_rw_224_sw-c228e218.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        crop_pct=0.95),
    'coatnet_rmlp_nano_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnet_rmlp_nano_rw_224_sw-bd1d51b3.pth',
        crop_pct=0.9),
    'coatnet_rmlp_0_rw_224.untrained': _cfg(url=''),
    'coatnet_rmlp_1_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnet_rmlp_1_rw_224_sw-9051e6c3.pth'),
    'coatnet_rmlp_2_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnet_rmlp_2_rw_224_sw-5ccfac55.pth'),
    'coatnet_rmlp_3_rw_224.untrained': _cfg(url=''),
    'coatnet_nano_cc_224.untrained': _cfg(url=''),
    'coatnext_nano_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/coatnext_nano_rw_224_ad-22cb71c2.pth',
        crop_pct=0.9),

    # ImagenNet-12k pretrain CoAtNet
    'coatnet_2_rw_224.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821),
    'coatnet_3_rw_224.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821),
    'coatnet_rmlp_1_rw2_224.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821),
    'coatnet_rmlp_2_rw_224.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821),

    # Trying to be like the CoAtNet paper configs (will adapt if 'tf' weights are ever released)
    'coatnet_0_224.untrained': _cfg(url=''),
    'coatnet_1_224.untrained': _cfg(url=''),
    'coatnet_2_224.untrained': _cfg(url=''),
    'coatnet_3_224.untrained': _cfg(url=''),
    'coatnet_4_224.untrained': _cfg(url=''),
    'coatnet_5_224.untrained': _cfg(url=''),

    # timm specific MaxVit configs, ImageNet-1k pretrain or untrained
    'maxvit_pico_rw_256.untrained': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxvit_nano_rw_256.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxvit_nano_rw_256_sw-fb127241.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxvit_tiny_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxvit_tiny_rw_224_sw-7d0dffeb.pth'),
    'maxvit_tiny_rw_256.untrained': _cfg(
        url='',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxvit_tiny_pm_256.untrained': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8)),

    # timm specific MaxVit w/ MLP rel-pos, ImageNet-1k pretrain
    'maxvit_rmlp_pico_rw_256.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxvit_rmlp_pico_rw_256_sw-8d82f2c6.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxvit_rmlp_nano_rw_256.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxvit_rmlp_nano_rw_256_sw-c17bb0d6.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxvit_rmlp_tiny_rw_256.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxvit_rmlp_small_rw_224.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth',
        crop_pct=0.9,
    ),
    'maxvit_rmlp_small_rw_256.untrained': _cfg(
        url='',
        input_size=(3, 256, 256), pool_size=(8, 8)),

    # timm specific MaxVit w/ ImageNet-12k pretrain and 1k fine-tune
    'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),

    # timm specific MaxVit w/ ImageNet-12k pretrain
    'maxvit_rmlp_base_rw_224.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
    ),

    # timm MaxxViT configs (ConvNeXt conv blocks mixed with MaxVit transformer blocks)
    'maxxvit_rmlp_nano_rw_256.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxxvit_rmlp_nano_rw_256_sw-0325d459.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxxvit_rmlp_tiny_rw_256.untrained': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxxvit_rmlp_small_rw_256.sw_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth',
        input_size=(3, 256, 256), pool_size=(8, 8)),

    # timm MaxxViT-V2 configs (ConvNeXt conv blocks mixed with MaxVit transformer blocks, more width, no block attn)
    'maxxvitv2_nano_rw_256.sw_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxxvitv2_rmlp_large_rw_224.untrained': _cfg(url=''),

    'maxxvitv2_rmlp_base_rw_224.sw_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821),

    # MaxViT models ported from official Tensorflow impl
    'maxvit_tiny_tf_224.in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_tiny_tf_384.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_tiny_tf_512.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    'maxvit_small_tf_224.in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_small_tf_384.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_small_tf_512.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    'maxvit_base_tf_224.in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_base_tf_384.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_base_tf_512.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    'maxvit_large_tf_224.in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_large_tf_384.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_large_tf_512.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),

    'maxvit_base_tf_224.in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843),
    'maxvit_base_tf_384.in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_base_tf_512.in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    'maxvit_large_tf_224.in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843),
    'maxvit_large_tf_384.in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_large_tf_512.in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), crop_pct=1.0, crop_mode='squash'),
    'maxvit_xlarge_tf_224.in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=21843),
    'maxvit_xlarge_tf_384.in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_xlarge_tf_512.in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    }
)

@register_model
def astroformer_5(pretrained=False, **kwargs) -> MaxxVit:
    return _create_maxxvit("astroformer_5", pretrained=pretrained, **kwargs)