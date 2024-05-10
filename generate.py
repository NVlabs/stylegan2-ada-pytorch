# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import subprocess
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
from numpy import linalg
import PIL.Image
import torch

import legacy

from opensimplex import OpenSimplex

# ---------------------------------------------------------------------------

class OSN():
    min = -1
    max = 1

    def __init__(self, seed, diameter):
        self.tmp = OpenSimplex(seed)
        self.d = diameter
        self.x = 0
        self.y = 0

    def get_val(self, angle):
        self.xoff = valmap(np.cos(angle), -1, 1, self.x, self.x + self.d)
        self.yoff = valmap(np.sin(angle), -1, 1, self.y, self.y + self.d)
        return self.tmp.noise2(self.xoff,self.yoff)

def circularloop(nf, d, seed, seeds):
    r = d/2

    zs = []
    # hardcoding in 512, prob TODO fix needed
    # latents_c = rnd.randn(1, G.input_shape[1])

    if(seeds is None):
        if seed:
            rnd = np.random.RandomState(seed)
        else:
            rnd = np.random
        latents_a = rnd.randn(1, 512)
        latents_b = rnd.randn(1, 512)
        latents_c = rnd.randn(1, 512)
    elif(len(seeds) is not 3):
        assert('Must choose exactly 3 seeds!')
    else:
        latents_a = np.random.RandomState(int(seeds[0])).randn(1, 512)
        latents_b = np.random.RandomState(int(seeds[1])).randn(1, 512)
        latents_c = np.random.RandomState(int(seeds[2])).randn(1, 512)

    latents = (latents_a, latents_b, latents_c)

    current_pos = 0.0
    step = 1./nf

    while(current_pos < 1.0):
        zs.append(circular_interpolation(r, latents, current_pos))
        current_pos += step
    return zs

def circular_interpolation(radius, latents_persistent, latents_interpolate):
    latents_a, latents_b, latents_c = latents_persistent

    latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

    latents_x = np.sin(np.pi * 2.0 * latents_interpolate) * radius
    latents_y = np.cos(np.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def size_range(s: str) -> List[int]:
    '''Accept a range 'a-c' and return as a list of 2 ints.'''
    return [int(v) for v in s.split('-')][::-1]

def line_interpolate(zs, steps, easing):
    out = []
    for i in range(len(zs)-1):
        for index in range(steps):
            t = index/float(steps)

            if(easing == 'linear'):
                out.append(zs[i+1]*t + zs[i]*(1-t))
            elif (easing == 'easeInOutQuad'):
                if(t < 0.5):
                    fr = 2 * t * t
                else:
                    fr = (-2 * t * t) + (4 * t) - 1
                out.append(zs[i+1]*fr + zs[i]*(1-fr))
            elif (easing == 'bounceEaseOut'):
                if (t < 4/11):
                    fr = 121 * t * t / 16
                elif (t < 8/11):
                    fr = (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
                elif t < 9/10:
                    fr = (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
                else:
                    fr = (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0
                out.append(zs[i+1]*fr + zs[i]*(1-fr))
            elif (easing == 'circularEaseOut'):
                fr = np.sqrt((2 - t) * t)
                out.append(zs[i+1]*fr + zs[i]*(1-fr))
            elif (easing == 'circularEaseOut2'):
                fr = np.sqrt(np.sqrt((2 - t) * t))
                out.append(zs[i+1]*fr + zs[i]*(1-fr))
            elif(easing == 'backEaseOut'):
                p = 1 - t
                fr = 1 - (p * p * p - p * math.sin(p * math.pi))
                out.append(zs[i+1]*fr + zs[i]*(1-fr))
    return out

def noiseloop(nf, d, seed):
    if seed:
        np.random.RandomState(seed)

    features = []
    zs = []
    for i in range(512):
      features.append(OSN(i+seed,d))

    inc = (np.pi*2)/nf
    for f in range(nf):
      z = np.random.randn(1, 512)
      for i in range(512):
        z[0,i] = features[i].get_val(inc*f)
      zs.append(z)

    return zs

def images(G,device,inputs,space,truncation_psi,label,noise_mode,outdir,start=None,stop=None):
    if(start is not None and stop is not None):
        tp = start
        tp_i = (stop-start)/len(inputs)

    for idx, i in enumerate(inputs):
        print('Generating image for frame %d/%d ...' % (idx, len(inputs)))
        
        if (space=='z'):
            z = torch.from_numpy(i).to(device)
            if(start is not None and stop is not None):
                img = G(z, label, truncation_psi=tp, noise_mode=noise_mode)
                tp = tp+tp_i
            else:
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        else:
            if len(i.shape) == 2: 
              i = torch.from_numpy(i).unsqueeze(0).to(device)
            img = G.synthesis(i, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/frame{idx:04d}.png')

def interpolate(G,device,projected_w,seeds,random_seed,space,truncation_psi,label,frames,noise_mode,outdir,interpolation,easing,diameter,start=None,stop=None):
    if(interpolation=='noiseloop' or interpolation=='circularloop'):
        if seeds is not None:
            print(f'Warning: interpolation type: "{interpolation}" doesn’t support set seeds.')

        if(interpolation=='noiseloop'):
            points = noiseloop(frames, diameter, random_seed)
        elif(interpolation=='circularloop'):
            points = circularloop(frames, diameter, random_seed, seeds)

    else:
        if projected_w is not None:
            points = np.load(projected_w)['w']
        else:
            # get zs from seeds
            points = seeds_to_zs(G,seeds)  
            # convert to ws
            if(space=='w'):
                points = zs_to_ws(G,device,label,truncation_psi,points)

        # get interpolation points
        if(interpolation=='linear'):
            points = line_interpolate(points,frames,easing)
        elif(interpolation=='slerp'):
            points = slerp_interpolate(points,frames)
            
    # generate frames
    images(G,device,points,space,truncation_psi,label,noise_mode,outdir,start,stop)

def seeds_to_zs(G,seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        z = np.random.RandomState(seed).randn(1, G.z_dim)
        zs.append(z)
    return zs

# slightly modified version of
# https://github.com/PDillis/stylegan2-fun/blob/master/run_generator.py#L399
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    v0 = v0.cpu().detach().numpy()
    v1 = v1.cpu().detach().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return torch.from_numpy(v2).to("cuda")

def slerp_interpolate(zs, steps):
    out = []
    for i in range(len(zs)-1):
        for index in range(steps):
            fraction = index/float(steps)
            out.append(slerp(fraction,zs[i],zs[i+1]))
    return out

def truncation_traversal(G,device,z,label,start,stop,increment,noise_mode,outdir):
    count = 1
    trunc = start

    z = seeds_to_zs(G,z)[0]
    z = torch.from_numpy(np.asarray(z)).to(device)

    while trunc <= stop:
        print('Generating truncation %0.2f' % trunc)
        
        img = G(z, label, truncation_psi=trunc, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/frame{count:04d}.png')

        trunc+=increment
        count+=1

def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

def zs_to_ws(G,device,label,truncation_psi,zs):
    ws = []
    for z_idx, z in enumerate(zs):
        z = torch.from_numpy(z).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=8)
        ws.append(w)
    return ws

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--diameter', type=float, help='diameter of loops', default=100.0, show_default=True)
@click.option('--frames', type=int, help='how many frames to produce (with seeds this is frames between each step, with loops this is total length)', default=240, show_default=True)
@click.option('--fps', type=int, help='framerate for video', default=24, show_default=True)
@click.option('--increment', type=float, help='truncation increment value', default=0.01, show_default=True)
@click.option('--interpolation', type=click.Choice(['linear', 'slerp', 'noiseloop', 'circularloop']), default='linear', help='interpolation type', required=True)
@click.option('--easing',
              type=click.Choice(['linear', 'easeInOutQuad', 'bounceEaseOut','circularEaseOut','circularEaseOut2']),
              default='linear', help='easing method', required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--process', type=click.Choice(['image', 'interpolation','truncation','interpolation-truncation']), default='image', help='generation method', required=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--random_seed', type=int, help='random seed value (used in noise and circular loop)', default=0, show_default=True)
@click.option('--scale-type',
                type=click.Choice(['pad', 'padside', 'symm','symmside']),
                default='pad', help='scaling method for --size', required=False)
@click.option('--size', type=size_range, help='size of output (in format x-y)')
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--space', type=click.Choice(['z', 'w']), default='z', help='latent space', required=True)
@click.option('--start', type=float, help='starting truncation value', default=0.0, show_default=True)
@click.option('--stop', type=float, help='stopping truncation value', default=1.0, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)

def generate_images(
    ctx: click.Context,
    easing: str,
    interpolation: str,
    increment: Optional[float],
    network_pkl: str,
    process: str,
    random_seed: Optional[int],
    diameter: Optional[float],
    scale_type: Optional[str],
    size: Optional[List[int]],
    seeds: Optional[List[int]],
    space: str,
    fps: Optional[int],
    frames: Optional[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    start: Optional[float],
    stop: Optional[float],
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    
    # custom size code from https://github.com/eps696/stylegan2ada/blob/master/src/_genSGAN2.py
    if(size): 
        print('render custom size: ',size)
        print('padding method:', scale_type )
        custom = True
    else:
        custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size 
    G_kwargs.scale_type = scale_type

    # mask/blend latents with external latmask or by splitting the frame
    latmask = False #temp
    if latmask is None:
        nHW = [int(s) for s in a.nXY.split('-')][::-1]
        assert len(nHW)==2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        # if a.verbose is True and n_mult > 1: print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = np.tile(np.asarray([[[[1]]]]), (1,n_mult,1,1))
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = a.splitfine
        lmask = torch.from_numpy(lmask).to(device)
    # else:
        # if a.verbose is True: print(' Latent blending with mask', a.latmask)
        # n_mult = 2
        # if os.path.isfile(a.latmask): # single file
        #     lmask = np.asarray([[img_read(a.latmask)[:,:,0] / 255.]]) # [h,w]
        # elif os.path.isdir(a.latmask): # directory with frame sequence
        #     lmask = np.asarray([[img_read(f)[:,:,0] / 255. for f in img_list(a.latmask)]]) # [h,w]
        # else:
        #     print(' !! Blending mask not found:', a.latmask); exit(1)
        # lmask = np.concatenate((lmask, 1 - lmask), 1) # [frm,2,h,w]
    # lmask = torch.from_numpy(lmask).to(device)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        # G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if (process=='image') and projected_w is not None:
        if seeds is not None:
            print ('Warning: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')


    if(process=='image'):
        if seeds is None:
            ctx.fail('--seeds option is required when not using --projected-w')

        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    elif(process=='interpolation' or process=='interpolation-truncation'):
        # create path for frames
        dirpath = os.path.join(outdir,'frames')
        os.makedirs(dirpath, exist_ok=True)

        # autogenerate video name: not great!
        if seeds is not None:
            seedstr = '_'.join([str(seed) for seed in seeds])
            vidname = f'{process}-{interpolation}-seeds_{seedstr}-{fps}fps'
        elif(interpolation=='noiseloop' or 'circularloop'):
            vidname = f'{process}-{interpolation}-{diameter}dia-seed_{random_seed}-{fps}fps'

        if process=='interpolation-truncation':
            interpolate(G,device,projected_w,seeds,random_seed,space,truncation_psi,label,frames,noise_mode,dirpath,interpolation,easing,diameter,start,stop)
        else:
            interpolate(G,device,projected_w,seeds,random_seed,space,truncation_psi,label,frames,noise_mode,dirpath,interpolation,easing,diameter)

        # convert to video
        cmd=f'ffmpeg -y -r {fps} -i {dirpath}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p {outdir}/{vidname}.mp4'
        subprocess.call(cmd, shell=True)

    elif(process=='truncation'):
        if seeds is None or (len(seeds)>1):
            ctx.fail('truncation requires a single seed value')

        # create path for frames
        dirpath = os.path.join(outdir,'frames')
        os.makedirs(dirpath, exist_ok=True)

        #vidname
        seed = seeds[0]
        vidname = f'{process}-seed_{seed}-start_{start}-stop_{stop}-inc_{increment}-{fps}fps'

        # generate frames
        truncation_traversal(G,device,seeds,label,start,stop,increment,noise_mode,dirpath)

        # convert to video
        cmd=f'ffmpeg -y -r {fps} -i {dirpath}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p {outdir}/{vidname}.mp4'
        subprocess.call(cmd, shell=True)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
