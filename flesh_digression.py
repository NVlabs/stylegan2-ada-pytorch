#
#   ~~ Flesh Digressions ~~
#         Or, Circular Interpolation of the StyleGAN Synthesis Network's Constant Layer
#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Based on halcy's circular interpolation script https://pastebin.com/RTtV2UY7

# Pytorch port

import argparse
import math
from datetime import datetime
from typing import Optional, Tuple, Union, List

import moviepy.editor
import numpy as np
import torch
from numpy import linalg

import dnnlib
import legacy


def circular_interpolation(radius: float, latents_persistent: Tuple[np.ndarray, np.ndarray, np.ndarray], circle_pos: float) -> np.ndarray:

    latents_a, latents_b, latents_c = latents_persistent

    latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

    latents_x = math.sin(math.pi * 2.0 * circle_pos) * radius
    latents_y = math.cos(math.pi * 2.0 * circle_pos) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents


def image_from_latent(G: torch.nn.Module, psi: float, z: np.ndarray, device: torch.device) -> np.ndarray:
    """Helper to genereate numpy array in RGB from numpy Z space vector"""
    z_tensor = torch.from_numpy(z).to(device)
    img = G(z_tensor, None, truncation_psi = psi, noise_mode = "const")
    # Convert NCHW to NHWC and cast to uint8
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img[0].cpu().numpy()

def size_range(s: str) -> List[int]:
    '''Accept a range 'a-c' and return as a list of 2 ints.'''
    return [int(v) for v in s.split('-')][::-1]

def seed_values(s: str) -> List[int]:
    '''Accept seeds 'a,b,c' and return as a list of 3 ints.'''
    return [int(v) for v in s.split(',')]

def generate_from_generator_adaptive(psi: float, radius_large: float, radius_small: float, step1: float, step2:float, video_length: float, seed: Optional[int], seeds: Optional[List], G: torch.nn.Module, device: torch.device):
    # psi = args.psi # 0.7
    # radius_large = args.radius_large # 600.0
    # radius_small = args.radius_small # 40.0
    current_position_increment = step1 # 0.005
    current_position_style_increment = step2 # 0.0025
    # video_length = args.video_length # 1.0
    # output_format = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    # latents for the circular interpolation in latent space
    if seed:
      rnd = np.random.RandomState(seed)
    else:
      rnd = np.random
    if seeds is None:
      latents_a = rnd.randn(1, G.z_dim)
      latents_b = rnd.randn(1, G.z_dim)
      latents_c = rnd.randn(1, G.z_dim)
    else:
      if(len(seeds) is not 3):
        print('you must set 3 seed values!')

      print(seeds)
      latents_a = np.random.RandomState(int(seeds[0])).randn(1, G.z_dim)
      latents_b = np.random.RandomState(int(seeds[1])).randn(1, G.z_dim)
      latents_c = np.random.RandomState(int(seeds[2])).randn(1, G.z_dim)

    latents_persistent_small = (latents_a, latents_b, latents_c)

    # latents for the circular interpolation of the unrolled constant layer
    latent_size = G.z_dim # latent z space size
    constant_layer_size = 4 # default StyleGAN constant layer size is 4x4
    # const_layer_total = 8192 
    constant_layer_total = G.synthesis.b4.const.data.flatten().size()[0] # type: ignore
    latents_aa = rnd.randn(1, constant_layer_total)
    latents_bb = rnd.randn(1, constant_layer_total)
    latents_cc = rnd.randn(1, constant_layer_total)
    latents_persistent_large = (latents_aa, latents_bb, latents_cc)

    # initialize the circular interpolation
    current_position = 0.0
    current_position_style = 0.0
    current_latent = circular_interpolation(radius_small, latents_persistent_small, current_position)
    current_image = image_from_latent(G, psi, current_latent, device)
    output_frames = []

    # Create the frames while interpolating along the circle, in both the latent space and the constant layer
    while(current_position_style < video_length):

        current_position += current_position_increment
        current_position_style += current_position_style_increment

        # interpolate the weights of the constant layer
        w = next(layer for name, layer in G.named_parameters() if name == 'synthesis.b4.const')
        # make a copy of the orig constant layer weights
        v1 = w.detach().clone()
        # unroll the constant layer
        v2 = v1.clone().reshape(1, constant_layer_total)
        with torch.no_grad():
            v2 += torch.from_numpy(circular_interpolation(radius_large, latents_persistent_large, current_position + np.pi)).to(device)
        v2 = v2.reshape(G.synthesis.b4.const.data.size()) # type: ignore
        G.synthesis.b4.const.copy_(v2) # type: ignore

        # interpolate along the latent space
        current_latent = circular_interpolation(radius_small, latents_persistent_small, current_position_style)
        current_image = image_from_latent(G, psi, current_latent, device)
        output_frames.append(current_image)

        G.synthesis.b4.const.copy_(v1) # type: ignore

        # stops at 1.0 (or whatever value to which video_length is set)
        print('Stopping at',video_length,'currently at',current_position_style, flush=True) 

    return output_frames

def main(pkl: str, psi: float, radius_large: float, radius_small:float, step1: float, step2: float, seed: Optional[int], video_length: float=1.0, size: int=None, seeds: int=None, scale_type: str='pad'):

    if(size): 
        print('render custom size: ',size)
        print('padding method:', scale_type )
        custom = True
    else:
        custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size 
    G_kwargs.scale_type = scale_type
    
    print('Loading networks from "%s"...' % pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(pkl) as f:
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device) # type: ignore

    frames = generate_from_generator_adaptive(psi,radius_large,radius_small,step1,step2,video_length,seed,seeds,G,device)
    frames = moviepy.editor.ImageSequenceClip(frames, fps=30)

    # Generate video at the current date and timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y-%I-%M-%S-%p")
    mp4_file = './circular-'+timestamp+'.mp4'
    mp4_codec = 'libx264'
    mp4_bitrate = '15M'
    mp4_fps = 24 # 20

    frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Creates a video of a circular interpolation of the constant layer for an input StyleGAN model.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pkl', help='A .pkl of a StyleGAN network model', required=True)
    parser.add_argument('--psi', help='The truncation psi used in the generator', default=0.7, type=float)
    parser.add_argument('--radius_large', help='The radius for the constant layer interpolation', default=300.0, type=float)
    parser.add_argument('--radius_small', help='The radius for the latent space interpolation', default=40.0, type=float)
    parser.add_argument('--step1', help='The value of the step/increment for the constant layer interpolation', default=0.005, type=float)
    parser.add_argument('--step2', help='The value of the step/increment for the latent space interpolation', default=0.0025, type=float)
    parser.add_argument('--seed', help='Seed value for random state', default=None, type=int)
    parser.add_argument('--seeds', help='Three comma separated seed values for circluar interpolation', default=None, type=seed_values)
    parser.add_argument('--size', help='Size of output (in format x-y)', default=None, type=size_range)
    parser.add_argument('--scale_type', help='Options: pad, padside, symm, symmside', default='pad', type=str)
    parser.add_argument('--video_length', help='The length of the video in terms of circular interpolation (recommended to keep at 1.0)', default=1.0, type=float)

    args = parser.parse_args()

    print(args.seeds)

    main(args.pkl, args.psi, args.radius_large, args.radius_small, args.step1, args.step2, args.seed, args.video_length, args.size, args.seeds, args.scale_type)
