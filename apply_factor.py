import os
import re
import subprocess
import argparse
import torch
from torchvision import utils # assumes you use torchvision 0.8.2; if you use the latest version, see comments below
import legacy
import dnnlib
from typing import List
import numpy as np
import random

"""
Use closed_form_factorization.py first to create your factor.pt

Usage:

python apply_factor.py -i 1-3 --seeds 10,20 --ckpt models/ffhq.pkl factor.pt --video
Create images and interpolation videos for image-seeds 10 and 20 for eigenvalues one, two and three.

python apply_factor.py -i 10,20 --seeds 100-200 --ckpt models/ffhq.pkl factor.pt --no-video
Create images for each image-seed between 100 and 200 and for eigenvalues 10 and 20.

python apply_factor.py --seeds r3 --ckpt models/ffhq.pkl factor.pt --no-video
Create images for three random seeds and all eigenvalues (this can take a lot of time, especially for videos).

Apply different truncation values by using --truncation.
Apply different increment degree for interpolation video by using --vid_increment.
Apply different scalar factors for moving latent vectors along eigenvector by using --degree.
Change output directory by using --output.
"""

#############################################################################################

def generate_images(z, label, truncation_psi, noise_mode, direction, file_name):
    if(args.space == 'w'):
        ws = zs_to_ws(G,torch.device('cuda'),label,truncation_psi,[z,z + direction,z - direction])
        img1 = G.synthesis(ws[0], noise_mode=noise_mode, force_fp32=True)
        img2 = G.synthesis(ws[1], noise_mode=noise_mode, force_fp32=True)
        img3 = G.synthesis(ws[2], noise_mode=noise_mode, force_fp32=True)
    else:
        img1 = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img2 = G(z + direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img3 = G(z - direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

    return torch.cat([img3, img1, img2], 0)

def generate_image(z, label, truncation_psi, noise_mode, space):
    if(space == 'w'):
        img = G.synthesis(z, noise_mode=noise_mode, force_fp32=True)
    else:
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    return img

def line_interpolate(zs, steps):
    out = []
    for i in range(len(zs)-1):
        for index in range(steps):
            t = index/float(steps)
            out.append(zs[i+1]*t + zs[i]*(1-t))
    return out

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c', a range 'a-c' and return as a list of ints or a string with "r{number}".'''
    if "r" in s:
        index = s.index("r")
        return int(s[index+1:])
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def zs_to_ws(G,device,label,truncation_psi,zs):
    ws = []
    for z_idx, z in enumerate(zs):
        # z = torch.from_numpy(z).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=8)
        ws.append(w)
    return ws

#############################################################################################

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Apply closed form factorization")
    parser.add_argument("-i", "--index", type=num_range, default="-1", help="index of eigenvector")
    parser.add_argument("--seeds", type=num_range, default="r1", help="list of random seeds or 'r10' for 10 random samples" )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument("--output", type=str, default="/cff_output/", help="directory for result samples",)
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2-ada-pytorch checkpoints")
    parser.add_argument("--space", type=str, default='w', help="generate images in the w space or z space")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation factor")
    parser.add_argument("factor", type=str, help="name of the closed form factorization result factor file")
    parser.add_argument("--vid_increment", type=float, default=0.1, help="increment degree for interpolation video")

    vid_parser = parser.add_mutually_exclusive_group(required=False)
    vid_parser.add_argument('--video', dest='vid', action='store_true')
    vid_parser.add_argument('--no-video', dest='vid', action='store_false')
    vid_parser.set_defaults(vid=False)

    args = parser.parse_args()

    device = torch.device('cuda')
    eigvec = torch.load(args.factor)["eigvec"].to(device)
    index = args.index
    seeds = args.seeds

    custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = None 
    G_kwargs.scale_type = 'symm'
    
    print('Loading networks from "%s"...' % args.ckpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)['G_ema'].to(device) # type: ignore


    if not os.path.exists(args.output):
      os.makedirs(args.output)

    label = torch.zeros([1, G.c_dim], device=device) # assume no class label
    noise_mode = "const" # default
    truncation_psi = args.truncation

    latents = []
    mode = "random"
    log_str = ""

    index_list_of_eigenvalues = []

    if isinstance(seeds, int):
        for i in range(seeds):
            latents.append(random.randint(0,2**32-1)) # 2**32-1 is the highest seed value
        mode = "random"
        log_str = str(seeds) + " samples"
    else:
        latents = seeds
        mode = "seeds"
        log_str = str(seeds)

    print(f"""
    Checkpoint: {args.ckpt}
    Factor: {args.factor}
    Outpur Directory: {args.output}
    Mode: {mode} ({log_str})
    Index: eigenvectors {index}
    Truncation: {truncation_psi}
    Video: {args.vid}
    Video Increments: {args.vid_increment}
    """)

    for l in latents:
        print(f"Generate images for seed ", l)

        z = torch.from_numpy(np.random.RandomState(l).randn(1, G.z_dim)).to(device)

        file_name = ""
        image_grid_eigvec = []

        if len(index) ==  1 and index[0] == -1: # use all eigenvalues
            index_list_of_eigenvalues = [*range(len(eigvec))]
            file_name = f"seed-{l}_index-all_degree-{args.degree}.png"
        else: # use certain indexes as eigenvalues
            index_list_of_eigenvalues = index
            str_index_list = '-'.join(str(x) for x in index)
            file_name = f"seed-{l}_index-{str_index_list}_degree-{args.degree}.png"

        for j in index_list_of_eigenvalues:
            current_eigvec = eigvec[:, j].unsqueeze(0)
            direction = args.degree * current_eigvec
            image_group = generate_images(z, label, truncation_psi, noise_mode, direction, file_name)
            image_grid_eigvec.append(image_group)

        print("Saving image ", os.path.join(args.output, file_name))
        grid = utils.save_image(
            torch.cat(image_grid_eigvec, 0),
            os.path.join(args.output, file_name),
            nrow = 3,
            normalize=True, 
            range=(-1, 1) # change range to value_range for latest torchvision
        )
        
    if(args.vid):
        print('Processing videos; this may take a while...')

        str_seed_list = '-'.join(str(x) for x in latents)
        str_index_list = '-'.join(str(x) for x in index_list_of_eigenvalues)

        folder_name = f"seed-{str_seed_list}_index-{str_index_list}_degree-{args.degree}"
        folder_path = os.path.join(args.output, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for l in latents:
            seed_folder_name = f"seed-{l}"
            seed_folder_path = os.path.join(folder_path, seed_folder_name)

            if not os.path.exists(seed_folder_path):
                os.makedirs(seed_folder_path)

            z = torch.from_numpy(np.random.RandomState(l).randn(1, G.z_dim)).to(device)


            for j in index_list_of_eigenvalues:
                current_eigvec = eigvec[:, j].unsqueeze(0)
                direction = args.degree * current_eigvec

                index_folder_name = f"index-{j}/frames"
                index_folder_path = os.path.join(seed_folder_path, index_folder_name)

                if not os.path.exists(index_folder_path):
                    os.makedirs(index_folder_path)

                if(args.space=='w'):
                    zs = [z-direction, z+direction]
                    ws = zs_to_ws(G,device,label,truncation_psi,zs)
                    pts = line_interpolate(ws, int((args.degree*2)/args.vid_increment))
                else:
                    pts = line_interpolate([z-direction, z+direction], int((args.degree*2)/args.vid_increment))
                
                fcount = 0
                for pt in pts:
                    img = generate_image(pt, label, truncation_psi, noise_mode, args.space)
                    grid = utils.save_image(
                        img,
                        f"{index_folder_path}/{fcount:04}.png",
                        normalize=True,
                        range=(-1, 1), # change range to value_range for latest torchvision
                        nrow=1,
                    )
                    fcount+=1
                cmd=f"ffmpeg -y -r 24 -i {index_folder_path}/%04d.png -vcodec libx264 -pix_fmt yuv420p {seed_folder_path}/seed-{str_seed_list}_index-{j}_degree-{args.degree}.mp4"
                subprocess.call(cmd, shell=True)

