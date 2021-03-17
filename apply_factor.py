import os
import subprocess
import argparse
import torch
from torchvision import utils
import legacy
import dnnlib
# import PIL.Image

def generate_images(z, label, truncation_psi, noise_mode, direction, file_name):
    img1 = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img2 = G(z + direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img3 = G(z - direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    return torch.cat([img3, img1, img2], 0)

def generate_image(z, label, truncation_psi, noise_mode, direction, file_name):
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    return img

def line_interpolate(zs, steps):
   out = []
   for i in range(len(zs)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
   return out

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Apply closed form factorization")
    parser.add_argument("-i", "--index", type=str, default="all", help="index of eigenvector")
    parser.add_argument("-s", "--samples", type=int, default=1, help="number of samples")
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument("--out_prefix", type=str, default="factor", help="filename prefix to result samples",)
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2-ada-pytorch checkpoints")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation factor")
    parser.add_argument( "factor", type=str, help="name of the closed form factorization result factor file")
    parser.add_argument("--vid_increment", type=float, default=0.1, help="increment degree for interpolation video")
    
    vid_parser = parser.add_mutually_exclusive_group(required=False)
    vid_parser.add_argument('--video', dest='vid', action='store_true')
    vid_parser.add_argument('--no-video', dest='vid', action='store_false')
    vid_parser.set_defaults(vid=False)

    args = parser.parse_args()
    device = torch.device('cuda')
    eigvec = torch.load(args.factor)["eigvec"].to(device)
    index = args.index
    if index != "all":
        try:
            index = int(index)
            if index > len(eigvec) - 1:
                raise IndexError("Index out of range; i > " + str(len(eigvec)))
        except ValueError:
            raise ValueError("Index must be 'all' or an int.") from None
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    image_grid = []
    file_name = f"sample-{args.samples}_index-{index}_degree-{args.degree}.png"
    
    latents = torch.randn([args.samples, G.z_dim]).cuda()
    label = torch.zeros([1, G.c_dim], device=device)
    noise_mode = "const" # default
    truncation_psi = args.truncation

    for i,l in enumerate(latents):
        z = l.unsqueeze(0)
        if index == "all":
            image_grid_eigvec = []
            file_name = f"sample-{i}_index-{index}_degree-{args.degree}.png"
            for j in range(len(eigvec)):
                current_eigvec = eigvec[:, j].unsqueeze(0)
                direction = args.degree * current_eigvec
                image_group = generate_images(z, label, truncation_psi, noise_mode, direction, file_name)
                image_grid_eigvec.append(image_group)
            grid = utils.save_image(
                torch.cat(image_grid_eigvec, 0),
                file_name,
                nrow = 3,
                normalize=True, 
                value_range=(-1, 1)
            )
        else:
            fn = f"sample-{i}.png"
            direction = args.degree * eigvec[:, index].unsqueeze(0)
            image_group = generate_images(z, label, truncation_psi, noise_mode, direction, fn)
            image_grid.append(image_group)
    if len(image_grid) > 0:
        grid = utils.save_image(
            torch.cat(image_grid, 0),
            file_name,
            nrow = 3,
            normalize=True, 
            value_range=(-1, 1)
        )

    if(args.vid):
        print('processing videos; this may take a while...')
        count = 0
        for l in latents:
            fname = f"{args.out_prefix}_index-{args.index}_degree-{args.degree}_index-{count}"
            if not os.path.exists(fname):
                os.makedirs(fname)

            if not os.path.exists(fname + '/frames'):
                os.makedirs(fname + '/frames')

            zs = line_interpolate([l-direction, l+direction], int((args.degree*2)/args.vid_increment))

            fcount = 0
            for z in zs:
                # generate latent
                img = generate_image(z, label, truncation_psi, noise_mode, direction, file_name)

                # generate latent
                grid = utils.save_image(
                    img,
                    f"{fname}/frames/{fname}_{fcount:04}.png",
                    normalize=True,
                    value_range=(-1, 1),
                    nrow=1,
                )

                fcount+=1


            cmd=f"ffmpeg -y -r 24 -i {fname}/frames/{fname}_%04d.png -vcodec libx264 -pix_fmt yuv420p {fname}/{fname}.mp4"
            subprocess.call(cmd, shell=True)

            count+=1