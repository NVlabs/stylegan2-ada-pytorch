import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import dnnlib
import legacy

import PIL
from PIL import Image

import numpy as np

import argparse
import copy
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_before', required=True, help='image that already has the W')
    parser.add_argument('--sample_after', required=True, help='image that does not inclue the W')
    parser.add_argument('--target_before', required=True, help='image that you want to apply W')
    parser.add_argument('--target_after', required=True, help='path of saving result')
    parser.add_argument('--network', required=True, help='pkl - url address')
    return vars(parser.parse_args())

def run(**kwargs):
    sample_before = kwargs['sample_before']
    sample_after = kwargs['sample_after']
    target_before = kwargs['target_before']
    target_after = kwargs['target_after']
    

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) 

    # install plug-in
    z = torch.randn([1, G.z_dim]).cuda()
    c = None
    img = G(z,c) 


    before = sample_before
    after = sample_after

    w_before = projection(before, 'Type-SB(Sample|Before)')
    w_after = projection(after, 'Type-SA(Sample|After)')

    w = w_after - w_before # age vector
    torch.save(w, 'get_w.pt')

    target = target_before

    w_target_before = projection(target, 'Type-TB(Target|Before)')
    w_target_after = w_target_before + w
    gen_target_after = generation(w_target_after,G) 
    img = Image.fromarray(gen_target_after)
    img.save(target_after)


def projection(img_path, id):
    id = id
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    s = min(w,h)

    #------------------------------#    
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) 
    #------------------------------#
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    img_uint8 = np.array(img, dtype=np.uint8)
        
    G_eval = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute w stats
    z_samples = np.random.randn(10000, G_eval.z_dim) # G_eval.z_dim == 512, (10000,512)
    w_samples = G_eval.mapping(torch.from_numpy(z_samples).to(device), None)
    w_samples = w_samples[:,:1,:].cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True) 
    w_std = (np.sum((w_samples - w_avg)**2)/10000)**0.5

    # Setup noise inputs
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
        
    # Load VGG16 feature detector
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    
    # Extract features for target image   

    img_tensor = torch.tensor(img_uint8.transpose([2,0,1]), device=device)
    img_tensor = img_tensor.unsqueeze(0).to(device).to(torch.float32)
    if img_tensor.shape[2] > 256:
        img_tensor = F.interpolate(img_tensor, size=(256,256), mode='area') # Resize to pass through the vgg16 network.
    img_features = vgg16(img_tensor, resize_images=False, return_lpips=True)
    # Set optimizer and Initiate noise
    num_steps = 1000
    initial_learning_rate = 0.1
    # ========================================= #

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    
    # projection
    num_steps = 1000
    lr_rampdown_length = 0.25
    lr_rampup_length = 0.05
    initial_noise_factor = 0.05
    noise_ramp_length = 0.75
    regularize_noise_weight = 1e5
    # ========================================= #

    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        # Synthesize image from opt_w
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')
        
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
            
        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (img_features - synth_features).square().sum() # Calculate the difference between two feature maps (target vs synth) generated through vgg. 
                                                                # This is the point of projection.
        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight
        
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step==0:
            print('[{}] projection start - Reproducing the image.. '.format(id))
        elif (step+1)%100 == 0 and (step+1) != num_steps:
            print(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        elif (step+1) == num_steps:
            print('projection clear')
        
        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    
    projected_w_steps = w_out.repeat([1, G.mapping.num_ws, 1])
    projected_w = projected_w_steps[-1]
    return projected_w

def generation(w,G):
    synth_image = G.synthesis(w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()    
    print('Finished creating a new image with the style(W) applied')
    return synth_image



if __name__ == '__main__':
    args = parse_command_line_args()
    network_pkl = args.network
    device = torch.device('cuda')
    run(**args)