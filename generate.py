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

#----------------------------------------------------------------------------

class OSN():
    min=-1
    max= 1

    def __init__(self,seed,diameter):
        self.tmp = OpenSimplex(seed)
        self.d = diameter
        self.x = 0
        self.y = 0

    def get_val(self,angle):
        self.xoff = valmap(np.cos(angle), -1, 1, self.x, self.x + self.d);
        self.yoff = valmap(np.sin(angle), -1, 1, self.y, self.y + self.d);
        return self.tmp.noise2d(self.xoff,self.yoff)

def circularloop(nf, d, seed):
    if seed:
        np.random.RandomState(seed)

    r = d/2

    zs = []

    rnd = np.random
    # hardcoding in 512, prob TODO fix needed
    # latents_c = rnd.randn(1, G.input_shape[1])
    latents_a = rnd.randn(1, 512)
    latents_b = rnd.randn(1, 512)
    latents_c = rnd.randn(1, 512)
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

def line_interpolate(zs, steps):
    out = []
    for i in range(len(zs)-1):
        for index in range(steps):
            fraction = index/float(steps)
            out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
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

def images(G,device,inputs,space,truncation_psi,label,noise_mode,outdir):
    for idx, i in enumerate(inputs):
        print('Generating image for frame %d/%d ...' % (idx, len(inputs)))
        
        if (space=='z'):
            z = torch.from_numpy(i).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        else:
            img = G.synthesis(i, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/frame{idx:04d}.png')

def interpolate(G,device,seeds,random_seed,space,truncation_psi,label,frames,noise_mode,outdir,interpolation,diameter):
    if(interpolation=='noiseloop' or interpolation=='circularloop'):
        if seeds is not None:
            print(f'Warning: interpolation type: "{interpolation}" doesn’t support set seeds.')

        if(interpolation=='noiseloop'):
            points = noiseloop(frames, diameter, random_seed)
        elif(interpolation=='circularloop'):
            points = circularloop(frames, diameter, random_seed)

    else:
        # get zs from seeds
        points = seeds_to_zs(G,seeds)
            
        # convert to ws
        if(space=='w'):
            points = zs_to_ws(G,device,label,truncation_psi,points)

        # get interpolation points
        if(interpolation=='linear'):
            points = line_interpolate(points,frames)
        elif(interpolation=='slerp'):
            if(space=='w'):
                print(f'Slerp currently isn’t supported in w space.')
            else:
                points = slerp_interpolate(points,frames)

    # generate frames
    images(G,device,points,space,truncation_psi,label,noise_mode,outdir)

def seeds_to_zs(G,seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        z = np.random.RandomState(seed).randn(1, G.z_dim)
        zs.append(z)
    return zs

# very hacky implementation of:
# https://github.com/soumith/dcgan.torch/issues/14
def slerp(val, low, high):
    assert low.shape == high.shape

    # z space
    if len(low.shape) == 2:
        out = np.zeros([low.shape[0],low.shape[1]])
        for i in range(low.shape[0]):
            omega = np.arccos(np.clip(np.dot(low[i]/np.linalg.norm(low[i]), high[i]/np.linalg.norm(high[i])), -1, 1))
            so = np.sin(omega)
            if so == 0:
                out[i] = (1.0-val) * low[i] + val * high[i] # L'Hopital's rule/LERP
            out[i] = np.sin((1.0-val)*omega) / so * low[i] + np.sin(val*omega) / so * high[i]
    # w space
    else:
        out = np.zeros([low.shape[0],low.shape[1],low.shape[2]])

        for i in range(low.shape[1]):
            omega = np.arccos(np.clip(np.dot(low[0][i]/np.linalg.norm(low[0][i]), high[0][i]/np.linalg.norm(high[0][i])), -1, 1))
            so = np.sin(omega)
            if so == 0:
                out[i] = (1.0-val) * low[0][i] + val * high[0][i] # L'Hopital's rule/LERP
            out[0][i] = np.sin((1.0-val)*omega) / so * low[0][i] + np.sin(val*omega) / so * high[0][i]

    return out

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
        w = G.mapping(z, label, truncation_psi=0.5, truncation_cutoff=8)
        ws.append(w)
    return ws

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--diameter', type=float, help='diameter of loops', default=100.0, show_default=True)
@click.option('--frames', type=int, help='how many frames to produce (with seeds this is frames between each step, with loops this is total length)', default=240, show_default=True)
@click.option('--fps', type=int, help='framerate for video', default=24, show_default=True)
@click.option('--increment', type=float, help='truncation increment value', default=0.01, show_default=True)
@click.option('--interpolation', type=click.Choice(['linear', 'slerp', 'noiseloop', 'circularloop']), default='linear', help='interpolation type', required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--process', type=click.Choice(['image', 'interpolation','truncation']), default='image', help='generation method', required=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--random_seed', type=int, help='random seed value (used in noise and circular loop)', default=0, show_default=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--space', type=click.Choice(['z', 'w']), default='z', help='latent space', required=True)
@click.option('--start', type=float, help='starting truncation value', default=0.0, show_default=True)
@click.option('--stop', type=float, help='stopping truncation value', default=1.0, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)

def generate_images(
    ctx: click.Context,
    interpolation: str,
    increment: Optional[float],
    network_pkl: str,
    process: str,
    random_seed: Optional[int],
    diameter: Optional[float],
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

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
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

    elif(process=='interpolation'):
        # create path for frames
        dirpath = os.path.join(outdir,'frames')
        os.makedirs(dirpath, exist_ok=True)

        # autogenerate video name: not great!
        if seeds is not None:
            seedstr = '_'.join([str(seed) for seed in seeds])
            vidname = f'{process}-{interpolation}-seeds_{seedstr}-{fps}fps'
        elif(interpolation=='noiseloop' or 'circularloop'):
            vidname = f'{process}-{interpolation}-{diameter}dia-seed_{random_seed}-{fps}fps'

        interpolate(G,device,seeds,random_seed,space,truncation_psi,label,frames,noise_mode,dirpath,interpolation,diameter)

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
