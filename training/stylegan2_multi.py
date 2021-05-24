# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

from .networks import FullyConnectedLayer, Conv2dLayer, ToRGBLayer, MappingNetwork

from util.utilgan import hw_scales, fix_size, multimask

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
# !!! custom
    # latmask,                      # mask for split-frame latents blending
    countHW         = [1,1],      # frame split count by height,width
    splitfine       = 0.,         # frame split edge fineness (float from 0+)
    size            = None,       # custom size
    scale_type      = None,       # scaling way: fit, centr, side, pad, padside
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
# !!! custom size & multi latent blending
        if size is not None and up==2:
            x = fix_size(x, size, scale_type)
            # x = multimask(x, size, latmask, countHW, splitfine)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
# !!! custom size & multi latent blending
    if size is not None and up==2:
        x = fix_size(x, size, scale_type)
        # x = multimask(x, size, latmask, countHW, splitfine)
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
# !!! custom
        countHW         = [1,1],      # frame split count by height,width
        splitfine       = 0.,         # frame split edge fineness (float from 0+)
        size            = None,       # custom size
        scale_type      = None,       # scaling way: fit, centr, side, pad, padside
        init_res        = [4,4],      # Initial (minimal) resolution for progressive training
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.countHW = countHW # !!! custom
        self.splitfine = splitfine # !!! custom
        self.size = size # !!! custom
        self.scale_type = scale_type # !!! custom
        self.init_res = init_res # !!! custom
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
# !!! custom
            self.register_buffer('noise_const', torch.randn([resolution * init_res[0]//4, resolution * init_res[1]//4]))
            # self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

# !!! custom 
    # def forward(self, x, latmask, w, noise_mode='random', fused_modconv=True, gain=1):
    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        # misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
# !!! custom
            sz = self.size if self.up==2 and self.size is not None else x.shape[2:]
            noise = torch.randn([x.shape[0], 1, *sz], device=x.device) * self.noise_strength
            # noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
# !!! custom noise size
            noise_size = self.size if self.up==2 and self.size is not None and self.resolution > 4 else x.shape[2:]
            noise = fix_size(noise.unsqueeze(0).unsqueeze(0), noise_size, scale_type=self.scale_type)[0][0]

        # print(x.shape, noise.shape, self.size, self.up)

        flip_weight = (self.up == 1) # slightly faster
        # x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            # latmask=latmask, countHW=self.countHW, splitfine=self.splitfine, size=self.size, scale_type=self.scale_type, # !!! custom
            # padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            countHW=self.countHW, splitfine=self.splitfine, size=self.size, scale_type=self.scale_type, # !!! custom
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)


        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
# !!! custom
        size                = None,       # custom size
        scale_type          = None,       # scaling way: fit, centr, side, pad, padside
        init_res            = [4,4],      # Initial (minimal) resolution for progressive training
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.size = size # !!! custom
        self.scale_type = scale_type # !!! custom
        self.init_res = init_res # !!! custom
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
# !!! custom
            self.const = torch.nn.Parameter(torch.randn([out_channels, *init_res]))
            # self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2, 
                init_res=init_res, scale_type=scale_type, size=size, # !!! custom
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution, 
            init_res=init_res, scale_type=scale_type, size=size, # !!! custom
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

# !!! custom
    # def forward(self, x, img, ws, latmask, dconst, force_fp32=False, fused_modconv=None, **layer_kwargs):
    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
# !!! custom const size
            if 'side' in self.scale_type and 'symm' in self.scale_type: # looks better
                const_size = self.init_res if self.size is None else self.size
                x = fix_size(x, const_size, self.scale_type)
# distortion technique from Aydao
            # x += dconst
        else:
            # misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
# !!! custom latmask
            # x = self.conv1(x, None, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
# !!! custom latmask
            # x = self.conv0(x, latmask, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            # x = self.conv1(x, None, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
# !!! custom latmask
            # x = self.conv0(x, latmask, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            # x = self.conv1(x, None, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
# !!! custom img size
            # misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
            img = fix_size(img, self.size, scale_type=self.scale_type)
            
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
# !!! custom
        init_res        = [4,4],      # Initial (minimal) resolution for progressive training
        size            = None,       # Output size
        scale_type      = None,       # scaling way: fit, centr, side, pad, padside
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        verbose         = False,      #
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.res_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.fmap_base = channel_base
        self.block_resolutions = [2 ** i for i in range(2, self.res_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.res_log2 + 1 - num_fp16_res), 8)

        # calculate intermediate layers sizes for arbitrary output resolution
        custom_res = (img_resolution * init_res[0] // 4, img_resolution * init_res[1] // 4)
        if size is None: size = custom_res
        if init_res != [4,4] and verbose:
            print(' .. init res', init_res, size)
        keep_first_layers = 2 if scale_type == 'fit' else None
        hws = hw_scales(size, custom_res, self.res_log2 - 2, keep_first_layers, verbose)
        if verbose: print(hws, '..', custom_res, self.res_log2-1)
        
        self.num_ws = 0
        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, 
                init_res=init_res, scale_type=scale_type, size=hws[i], # !!! custom
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    # def forward(self, ws, latmask, dconst, **block_kwargs):
    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
# !!! custom
            # x, img = block(x, img, cur_ws, latmask, dconst, **block_kwargs)
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
# !!! custom
        init_res            = [4,4],  # Initial (minimal) resolution for progressive training
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.init_res = init_res # !!! custom
        self.img_channels = img_channels
# !!! custom
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, init_res=init_res, img_channels=img_channels, **synthesis_kwargs) # !!! custom
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
# !!! custom
        self.output_shape = [1, img_channels, img_resolution * init_res[0] // 4, img_resolution * init_res[1] // 4]

# !!! custom
    # def forward(self, z, c, latmask, dconst, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
    # def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        # img = self.synthesis(ws, latmask, dconst, **synthesis_kwargs) # !!! custom
        img = self.synthesis(ws, **synthesis_kwargs) # !!! custom
        return img
