from typing import AnyStr, Tuple, List, Dict, Any, Optional
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import PIL
import numpy as np
from skimage import measure


import dnnlib
import legacy
from torch_utils import misc

from .fewshot import FewShotCNN


class VGG16Singleton:
    _instance = None
    vgg = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is  None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        
        return cls._instance

    def __init__(self) -> None:
        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg = torch.jit.load(f, map_location=torch.device("cuda")).eval().to("cuda")

class StyleGAN2Ada:
    def __init__(self, image_size: int, device: AnyStr, model_path: Optional[AnyStr] = None, num_projection_steps: int = 1000) -> None:
        if model_path is None or not Path(model_path).exists():
            raise ValueError(f"Invalid model path '{model_path}'")

        self.device = device
        self.num_projection_steps = num_projection_steps
        self._paper_256_cfg = dnnlib.EasyDict(ref_gpus=8, kimg=25000, mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1, ema=20,  ramp=None, map=8)
        self._common_kwargs = dict(
            c_dim=0,  # Number of classes -> We have one class c_dim = 0
            img_resolution=image_size,
            img_channels=3
        )

        self._gen_kwargs = dnnlib.EasyDict(
            class_name='training.networks.Generator',
            z_dim=512,
            w_dim=512,
            mapping_kwargs=dnnlib.EasyDict(),
            synthesis_kwargs=dnnlib.EasyDict())

        self._dis_kwargs = dnnlib.EasyDict(
            class_name='training.networks.Discriminator',
            block_kwargs=dnnlib.EasyDict(),
            mapping_kwargs=dnnlib.EasyDict(),
            epilogue_kwargs=dnnlib.EasyDict())

        self._gen_kwargs.synthesis_kwargs.channel_base = self._dis_kwargs.channel_base = int(
            self._paper_256_cfg.fmaps * 32768)
        self._gen_kwargs.synthesis_kwargs.channel_max = self._dis_kwargs.channel_max = 512
        self._gen_kwargs.synthesis_kwargs.num_fp16_res = self._dis_kwargs.num_fp16_res = 4  # enable mixed-precision training

        self._gen_kwargs.synthesis_kwargs.conv_clamp = self._dis_kwargs.conv_clamp = 256  # clamp activations to avoid float16 overflow
        self._gen_kwargs.mapping_kwargs.num_layers = self._paper_256_cfg.map

        self._dis_kwargs.epilogue_kwargs.mbstd_group_size = self._paper_256_cfg.mbstd

        self._gen_optim_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam',
                                                 lr=self._paper_256_cfg.lrate,
                                                 betas=[0, 0.99],
                                                 eps=1e-8)
        self._dis_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam',
                                               lr=self._paper_256_cfg.lrate,
                                               betas=[0, 0.99],
                                               eps=1e-8)

        # Load Model
        self.generator = dnnlib.util.construct_class_by_name(
            **self._gen_kwargs,
            **self._common_kwargs).eval().requires_grad_(False).to(
                self.device)  # subclass of torch.nn.Module
        self.discriminator = dnnlib.util.construct_class_by_name(
            **self._dis_kwargs,
            **self._common_kwargs).eval().requires_grad_(False).to(
                self.device)  # subclass of torch.nn.Module
        self.generator_copy = copy.deepcopy(self.generator).eval().to(self.device)

        with dnnlib.util.open_url(str(model_path)) as f:
            resume_data = legacy.load_network_pkl(f)
            for name, module in [('G', self.generator),
                                 ('D', self.discriminator),
                                 ('G_ema', self.generator_copy)]:
                print("loading: ", name)
                misc.copy_params_and_buffers(resume_data[name],
                                             module,
                                             require_all=False)

        self.vgg16 = VGG16Singleton().vgg

    def project_image(
        self,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
    ) -> torch.Tensor:
        z_samples = np.random.RandomState(123).randn(w_avg_samples,
                                                     self.generator.z_dim)
        w_samples = self.generator.mapping(
            torch.from_numpy(z_samples).to(self.device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(
            np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg)**2) / w_avg_samples)**0.5
        noise_bufs = {
            name: buf
            for (name, buf) in self.generator.synthesis.named_buffers()
            if 'noise_const' in name
        }

        # Features for target image.
        target_images = target.unsqueeze(0).to(self.device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images,
                                          size=(256, 256),
                                          mode='area')

        target_features = self.vgg16(target_images,
                                     resize_images=False,
                                     return_lpips=True)

        w_opt = torch.tensor(w_avg,
                             dtype=torch.float32,
                             device=self.device,
                             requires_grad=True)  # pylint: disable=not-callable
        w_out = torch.zeros([self.num_projection_steps] + list(w_opt.shape[1:]),
                            dtype=torch.float32,
                            device=self.device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()),
                                     betas=(0.9, 0.999),
                                     lr=initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf.requires_grad = False
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(self.num_projection_steps):
            # Learning rate schedule.
            t = step / self.num_projection_steps
            w_noise_scale = w_std * initial_noise_factor * max(
                0.0, 1.0 - t / noise_ramp_length)**2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat(
                [1, self.generator.mapping.num_ws, 1])
            synth_images = self.generator.synthesis(ws, noise_mode='const')

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255 / 2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images,
                                             size=(256, 256),
                                             mode='area')

            # Features for synth images.
            synth_features = self.vgg16(synth_images,
                                        resize_images=False,
                                        return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,
                          None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise *
                                 torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise *
                                 torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Save projected W for each optimization step.
            w_out[step] = w_opt.detach()[0]

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        return w_out.repeat([1, self.generator.mapping.num_ws, 1])

    def get_image_and_features(self, projected_w_steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            projected_w = projected_w_steps[-1]
            synth_image, features = self.generator.synthesis(
                projected_w.unsqueeze(0),
                noise_mode='const',
                extract_features=True)
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        return synth_image, features

class PartSegmentor:
    def __init__(self,
        stylegan2ada_config: Optional[Dict[AnyStr, Any]] = None,
        fewshot_config: Optional[Dict[AnyStr, Any]] = None, 
        classes: Optional[List[AnyStr]] = None, 
        image_size: int = 256, 
        device: str = "cuda"
    ) -> None:

        if stylegan2ada_config is None:
            raise ValueError("StyleGAN2ADA config missing")

        if fewshot_config is None:
            raise ValueError("FewShot config is missing")
        
        if classes is None or len(classes) == 0:
            raise ValueError("No classes specified")

        self.classes = classes
        self.image_size = image_size
        self.device = device
        self.stylegan2ada = StyleGAN2Ada(image_size, device, **stylegan2ada_config)
        self.few_shot: FewShotCNN = FewShotCNN(fewshot_config["in_channels"],
                                               fewshot_config["num_classes"],
                                               fewshot_config["size"])

        self.few_shot.load_state_dict(torch.load(fewshot_config["model_path"]))
        self.few_shot.eval()
        # self.few_shot.to(device)

    def segment_parts(self, region: PIL.Image, save_image=None) -> Tuple:
        processed_region: torch.Tensor = self.preprocess_image(region)

        projected_w_steps = self.stylegan2ada.project_image(processed_region)
        synth_image, features = self.stylegan2ada.get_image_and_features(projected_w_steps)

        if save_image is not None:  
            img = PIL.Image.fromarray(synth_image, 'RGB')
            img.save(f'projected/{save_image}_result.png', "PNG")
        
        predictions = self.get_predictions(features)
        
        feature_vector = self.get_feature_vector(predictions)
        return feature_vector
             
    def preprocess_image(self, image: PIL.Image) -> torch.Tensor:
        resized = self.resize_region(image, (256, 256))
        as_array = np.array(resized, dtype=np.uint8)
        tensored = torch.tensor(as_array.transpose([2, 0, 1]), device=self.device)

        return tensored

    def resize_region(self, image: PIL.Image, new_size: Tuple[int, int]) -> PIL.Image:
        return image.resize(new_size, PIL.Image.LANCZOS)

    def get_predictions(self, features) -> np.array:
        with torch.no_grad():
            torch.cuda.empty_cache()
            out = self.few_shot(self.concat_features(features))
            predictions: np.ndarray = out.data.max(1)[1].cpu().numpy()

        return predictions 

    def get_feature_vector(self, predictions: np.ndarray) -> Tuple:
        feature_vector = [0] * len(self.classes)
        
        for class_id, _ in enumerate(feature_vector):
            # Convert to 1/0 array
            new_arr = np.copy(predictions)
            new_arr[new_arr!=class_id] = 0
            # Label 
            label_image, num_regions = measure.label(new_arr, return_num=True)
            region_properties = measure.regionprops(label_image)
            num_regions = 0
            for region in region_properties:
                if region.area > 75:
                    num_regions += 1

            feature_vector[class_id] = num_regions

        return tuple(feature_vector)

    @torch.no_grad()
    def concat_features(self, features):
        h = max([f.shape[-2] for f in features])
        w = max([f.shape[-1] for f in features])
        return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)
