import os
import sys
import time
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline as CubSpline
from scipy.special import comb
import scipy
from imageio import imread

import torch
import torch.nn.functional as F

# from perlin import PerlinNoiseFactory as Perlin
# noise = Perlin(1)

# def latent_noise(t, dim, noise_step=78564.543):
    # latent = np.zeros((1, dim))
    # for i in range(dim):
        # latent[0][i] = noise(t + i * noise_step)
    # return latent

def load_latents(npy_file):
    key_latents = np.load(npy_file)
    try:
        key_latents = key_latents[key_latents.files[0]]
    except:
        pass
    idx_file = os.path.splitext(npy_file)[0] + '.txt'
    if os.path.exists(idx_file): 
        with open(idx_file) as f:
            lat_idx = f.readline()
            lat_idx = [int(l.strip()) for l in lat_idx.split(',') if '\n' not in l and len(l.strip())>0]
        key_latents = [key_latents[i] for i in lat_idx]
    return np.asarray(key_latents)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def get_z(shape, seed=None, uniform=False):
    if seed is None:
        seed = np.random.seed(int((time.time()%1) * 9999))
    rnd = np.random.RandomState(seed)
    if uniform:
        return rnd.uniform(0., 1., shape)
    else:
        return rnd.randn(*shape) # *x unpacks tuple/list to sequence

def smoothstep(x, NN=1., xmin=0., xmax=1.):
    N = math.ceil(NN)
    x = np.clip((x - xmin) / (xmax - xmin), 0, 1)
    result = 0
    for n in range(0, N+1):
         result += scipy.special.comb(N+n, n) * scipy.special.comb(2*N+1, N-n) * (-x)**n
    result *= x**(N+1)
    if NN != N: result = (x + result) / 2
    return result

def lerp(z1, z2, num_steps, smooth=0.): 
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interpol = z1 + (z2 - z1) * x
        vectors.append(interpol)
    return np.array(vectors)

# interpolate on hypersphere
def slerp(z1, z2, num_steps, smooth=0.):
    z1_norm = np.linalg.norm(z1)
    z2_norm = np.linalg.norm(z2)
    z2_normal = z2 * (z1_norm / z2_norm)
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interplain = z1 + (z2 - z1) * x
        interp = z1 + (z2_normal - z1) * x
        interp_norm = np.linalg.norm(interp)
        interpol_normal = interplain * (z1_norm / interp_norm)
        # interpol_normal = interp * (z1_norm / interp_norm)
        vectors.append(interpol_normal)
    return np.array(vectors)

def cublerp(points, steps, fstep):
    keys = np.array([i*fstep for i in range(steps)] + [steps*fstep])
    points = np.concatenate((points, np.expand_dims(points[0], 0)))
    cspline = CubSpline(keys, points)
    return cspline(range(steps*fstep+1))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
def latent_anima(shape, frames, transit, key_latents=None, smooth=0.5, cubic=False, gauss=False, seed=None, verbose=True):
    if key_latents is None:
        transit = int(max(1, min(frames//4, transit)))
    steps = max(1, int(frames // transit))
    log = ' timeline: %d steps by %d' % (steps, transit)

    getlat = lambda : get_z(shape, seed=seed)
    
    # make key points
    if key_latents is None:
        key_latents = np.array([getlat() for i in range(steps)])

    latents = np.expand_dims(key_latents[0], 0)
    
    # populate lerp between key points
    if transit == 1:
        latents = key_latents
    else:
        if cubic:
            latents = cublerp(key_latents, steps, transit)
            log += ', cubic'
        else:
            for i in range(steps):
                zA = key_latents[i]
                zB = key_latents[(i+1) % steps]
                interps_z = slerp(zA, zB, transit, smooth=smooth)
                latents = np.concatenate((latents, interps_z))
    latents = np.array(latents)
    
    if gauss:
        lats_post = gaussian_filter(latents, [transit, 0, 0], mode="wrap")
        lats_post = (lats_post / np.linalg.norm(lats_post, axis=-1, keepdims=True)) * math.sqrt(np.prod(shape))
        log += ', gauss'
        latents = lats_post
        
    if verbose: print(log)
    if latents.shape[0] > frames: # extra frame
        latents = latents[1:]
    return latents
    
# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
def multimask(x, size, latmask=None, countHW=[1,1], delta=0.):
    Hx, Wx = countHW
    bcount = x.shape[0]

    if max(countHW) > 1:
        W = x.shape[3] # width
        H = x.shape[2] # height
        if Wx > 1:
            stripe_mask = []
            for i in range(Wx):
                ch_mask = peak_roll(W, Wx, i, delta).unsqueeze(0).unsqueeze(0) # [1,1,w] th
                ch_mask = ch_mask.repeat(1,H,1) # [1,h,w]
                stripe_mask.append(ch_mask)
            maskW = torch.cat(stripe_mask, 0).unsqueeze(1) # [x,1,h,w]
        else: maskW = [1]
        if Hx > 1:
            stripe_mask = []
            for i in range(Hx):
                ch_mask = peak_roll(H, Hx, i, delta).unsqueeze(1).unsqueeze(0) # [1,h,1] th
                ch_mask = ch_mask.repeat(1,1,W) # [1,h,w]
                stripe_mask.append(ch_mask)
            maskH = torch.cat(stripe_mask, 0).unsqueeze(1) # [y,1,h,w]
        else: maskH = [1]

        mask = []
        for i in range(Wx):
            for j in range(Hx):
                mask.append(maskW[i] * maskH[j])
        mask = torch.cat(mask, 0).unsqueeze(1) # [xy,1,h,w]
        mask = mask.to(x.device)
        x = torch.sum(x[:Hx*Wx] * mask, 0, keepdim=True)

    elif latmask is not None:
        if len(latmask.shape) < 4:
            latmask = latmask.unsqueeze(1) # [b,1,h,w]
        lms = latmask.shape
        if list(lms[2:]) != list(size) and np.prod(lms) > 1:
            latmask = F.interpolate(latmask, size) # , mode='nearest'
        latmask = latmask.type(x.dtype)
        x = torch.sum(x[:lms[0]] * latmask, 0, keepdim=True)
    else:
        return x

    x = x.repeat(bcount,1,1,1)
    return x # [b,f,h,w]

def peak_roll(width, count, num, delta):
    step = width // count
    if width > step*2:
        fill_range = torch.zeros([width-step*2])
        full_ax = torch.cat((peak(step, delta), fill_range), 0)
    else:
        full_ax = peak(step, delta)[:width]
    if num == 0: 
        shift = max(width - (step//2), 0.) # must be positive!
    else:
        shift = step*num - (step//2)
    full_ax = torch.roll(full_ax, shift, 0)
    return full_ax # [width,]

def peak(steps, delta):
    x = torch.linspace(0.-delta, 1.+ delta, steps)
    x_rev = torch.flip(x,[0])
    x = torch.cat((x, x_rev), 0)
    x = torch.clip(x, 0., 1.)
    return x # [steps*2,]

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
def ups2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    s = x.shape
    x = x.reshape(-1, s[1], s[2], 1, s[3], 1)
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = x.reshape(-1, s[1], s[2] * factor, s[3] * factor)
    return x

# Tiles an array around two points, allowing for pad lengths greater than the input length
# NB: if symm=True, every second tile is mirrored = messed up in GAN
# adapted from https://discuss.pytorch.org/t/symmetric-padding/19866/3
def tile_pad(xt, padding, symm=True):
    h, w = xt.shape[-2:]
    left, right, top, bottom = padding
 
    def tile(x, minx, maxx, symm=True):
        rng = maxx - minx
        if symm is True: # triangular reflection
            double_rng = 2*rng
            mod = np.fmod(x - minx, double_rng)
            normed_mod = np.where(mod < 0, mod+double_rng, mod)
            out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        else: # repeating tiles
            mod = np.remainder(x - minx, rng)
            out = mod + minx
        return np.array(out, dtype=x.dtype)

    x_idx = np.arange(-left, w+right)
    y_idx = np.arange(-top, h+bottom)
    x_pad = tile(x_idx, -0.5, w-0.5, symm)
    y_pad = tile(y_idx, -0.5, h-0.5, symm)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return xt[..., yy, xx]

def pad_up_to(x, size, type='centr'):
    sh = x.shape[2:][::-1]
    if list(x.shape[2:]) == list(size): return x
    padding = []
    for i, s in enumerate(size[::-1]):
        if 'side' in type.lower():
            padding = padding + [0, s-sh[i]]
        else: # centr
            p0 = (s-sh[i]) // 2
            p1 = s-sh[i] - p0
            padding = padding + [p0,p1]
    y = tile_pad(x, padding, symm = 'symm' in type.lower())
    # if 'symm' in type.lower():
        # y = tile_pad(x, padding, symm=True)
    # else:
        # y = F.pad(x, padding, 'circular')
    return y

# scale_type may include pad, side, symm
def fix_size(x, size, scale_type='centr'): 
    if not len(x.shape) == 4:
        raise Exception(" Wrong data rank, shape:", x.shape)
    if x.shape[2:] == size:
        return x
    if (x.shape[2]*2, x.shape[3]*2) == size:
        return ups2d(x)

    if scale_type.lower() == 'fit':
        return F.interpolate(x, size, mode='nearest') # , align_corners=True
    elif 'pad' in scale_type.lower():
        pass
    else: # proportional scale to smaller side, then pad to bigger side
        sh0 = x.shape[2:]
        upsc = np.min(size) / np.min(sh0)
        new_size = [int(sh0[i]*upsc) for i in [0,1]]
        x = F.interpolate(x, new_size, mode='nearest') # , align_corners=True

    x = pad_up_to(x, size, scale_type)
    return x

# Make list of odd sizes for upsampling to arbitrary resolution
def hw_scales(size, base, n, keep_first_layers=None, verbose=False):
    if isinstance(base, int): base = (base, base)
    start_res = [int(b * 2 ** (-n)) for b in base]
    
    start_res[0] = int(start_res[0] * size[0] // base[0])
    start_res[1] = int(start_res[1] * size[1] // base[1])

    hw_list = []
    
    if base[0] != base[1] and verbose is True:
        print(' size', size, 'base', base, 'start_res', start_res, 'n', n)
    if keep_first_layers is not None and keep_first_layers > 0:
        for i in range(keep_first_layers):
            hw_list.append(start_res)
            start_res = [x*2 for x in start_res]
            n -= 1
            
    ch = (size[0] / start_res[0]) ** (1/n)
    cw = (size[1] / start_res[1]) ** (1/n)
    for i in range(n):
        h = math.floor(start_res[0] * ch**i)
        w = math.floor(start_res[1] * cw**i)
        hw_list.append((h,w))

    hw_list.append(size)
    return hw_list

def calc_res(shape):
    base0 = 2**int(np.log2(shape[0]))
    base1 = 2**int(np.log2(shape[1]))
    base = min(base0, base1)
    min_res = min(shape[0], shape[1])
    
    def int_log2(xs, base):
        return [x * 2**(2-int(np.log2(base))) % 1 == 0 for x in xs]
    if min_res != base or max(*shape) / min(*shape) >= 2:
        if np.log2(base) < 10 and all(int_log2(shape, base*2)):
            base = base * 2

    return base # , [shape[0]/base, shape[1]/base]

def calc_init_res(shape, resolution=None):
    if len(shape) == 1:
        shape = [shape[0], shape[0], 1]
    elif len(shape) == 2:
        shape = [*shape, 1]
    size = shape[:2] if shape[2] < min(*shape[:2]) else shape[1:] # fewer colors than pixels
    if resolution is None:
        resolution = calc_res(size)
    res_log2 = int(np.log2(resolution))
    init_res = [int(s * 2**(2-res_log2)) for s in size]
    return init_res, resolution, res_log2

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def file_list(path, ext=None, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    if ext is not None: 
        if isinstance(ext, list):
            files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ext]
        elif isinstance(ext, str):
            files = [f for f in files if f.endswith(ext)]
        else:
            print(' Unknown extension/type for file list!')
    return sorted([f for f in files if os.path.isfile(f)])

def dir_list(in_dir):
    dirs = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
    return sorted([f for f in dirs if os.path.isdir(f)])

def img_list(path, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['jpg', 'jpeg', 'png', 'ppm', 'tif']]
    return sorted([f for f in files if os.path.isfile(f)])

def img_read(path):
    img = imread(path)
    # 8bit to 256bit
    if (img.ndim == 2) or (img.shape[2] == 1):
        img = np.dstack((img,img,img))
    # rgba to rgb 
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img
    