from collections import namedtuple
from dataclasses import dataclass
from distutils.command.config import config
from typing import AnyStr, List, Tuple, Dict, Any
import copy
from pathlib import Path

from skimage import io, transform, measure
import numpy as np
import dnnlib
import legacy
import selectivesearch
import torch
import torch.nn.functional as F
import ruamel.yaml as yaml

from torch_utils import misc
import prediction_utils
from svm.svm import SVM
from fewshot import FewShotCNN


DEVICE = "cuda"


@dataclass
class ImageColletion:
    image: np.ndarray
    feature: np.ndarray
    latent: np.ndarray


ClassPrediction = namedtuple('ClassPrediction', ['prediction', 'region_id'])
BoundingBox = namedtuple('BoundingBox', ["x_tl", "y_tl", "x_br", "y_br"])


class StyleGAN2Ada:

    def __init__(self, model_path: AnyStr, image_size: int = 256) -> None:
        self._network = None
        # Setup Network Arguments - from projector.py
        self._paper_256_cfg = dnnlib.EasyDict(ref_gpus=8,
                                              kimg=25000,
                                              mb=64,
                                              mbstd=8,
                                              fmaps=0.5,
                                              lrate=0.0025,
                                              gamma=1,
                                              ema=20,
                                              ramp=None,
                                              map=8)
        self._common_kwargs = dict(
            c_dim=0,  # Number of classes -> We have one class c_dim = 0
            img_resolution=image_size,
            img_channels=3)

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
            **self._common_kwargs).train().requires_grad_(False).to(
                DEVICE)  # subclass of torch.nn.Module
        self.discriminator = dnnlib.util.construct_class_by_name(
            **self._dis_kwargs,
            **self._common_kwargs).train().requires_grad_(False).to(
                DEVICE)  # subclass of torch.nn.Module
        self.generator_copy = copy.deepcopy(self.generator).eval().to(DEVICE)

        with dnnlib.util.open_url(str(model_path)) as f:
            resume_data = legacy.load_network_pkl(f)
            for name, module in [('G', self.generator),
                                 ('D', self.discriminator),
                                 ('G_ema', self.generator_copy)]:
                print("loading: ", name)
                misc.copy_params_and_buffers(resume_data[name],
                                             module,
                                             require_all=False)

        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(DEVICE)

    def project_image(
        self,
        target: torch.
        Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
    ):
        z_samples = np.random.RandomState(123).randn(w_avg_samples,
                                                     self.generator.z_dim)
        w_samples = self.generator.mapping(
            torch.from_numpy(z_samples).to(DEVICE), None)  # [N, L, C]
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
        target_images = target.unsqueeze(0).to(DEVICE).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images,
                                          size=(256, 256),
                                          mode='area')
        target_features = self.vgg16(target_images,
                                     resize_images=False,
                                     return_lpips=True)

        w_opt = torch.tensor(w_avg,
                             dtype=torch.float32,
                             device=DEVICE,
                             requires_grad=True)  # pylint: disable=not-callable
        w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]),
                            dtype=torch.float32,
                            device=DEVICE)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()),
                                     betas=(0.9, 0.999),
                                     lr=initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(num_steps):
            # Learning rate schedule.
            t = step / num_steps
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

    def get_image_and_features(self, projected_w_steps: torch.Tensor):
        projected_w = projected_w_steps[-1]
        synth_image, features = self.generator.synthesis(
            projected_w.unsqueeze(0),
            noise_mode='const',
            extract_features=True)
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(
            torch.uint8)[0].cpu().numpy()
        # image = PIL.Image.fromarray(synth_image, 'RGB')

        return synth_image, features


class Predictor:

    def __init__(self, beast: AnyStr, projector_path: AnyStr,
                 fewshot_config: Dict[AnyStr, AnyStr], classes: List[AnyStr],
                 svm_path: AnyStr) -> None:
        self.beast = beast
        self.projector: StyleGAN2Ada = StyleGAN2Ada(projector_path)
        self.few_shot: FewShotCNN = FewShotCNN(fewshot_config["in_channels"],
                                               fewshot_config["num_classes"],
                                               fewshot_config["size"])

        self.few_shot.load_state_dict(torch.load(fewshot_config["model_path"]))
        self.classes: List[AnyStr] = classes
        self._feature_vector = namedtuple(f"{self.beast}FeatureVector",
                                          self.classes)

        self.svm: SVM = SVM()
        # self.svm.load_model(Path(svm_path))

    def predict_beast(self, image: np.ndarray) -> AnyStr:
        image = np.copy(image)

        suggested_regions = self.get_region_suggestions(image)

        processed_regions = self.process_regions(suggested_regions, image)

        projected_images = self.project_to_latent(processed_regions)

        features: List[self._feature_vector] = []
        for projected_image in projected_images:
            features.append(self.get_features(projected_image))

        predicted_classes = []
        for feature in features:
            predicted_classes.append(self.predict_class(feature))

        results = self.filter_predictions(suggested_regions, predicted_classes)

        return None

    def get_region_suggestions(self, image) -> List[BoundingBox]:
        _, result = selectivesearch.selective_search(image)

        boxes = [BoundingBox(*region["rect"]) for region in result.values()]

        return boxes

    def process_regions(self, region_suggestions, image) -> List:
        processed_regions = []
        for suggested_region in region_suggestions:
            cropped_region = self.crop_image(image, suggested_region)

            resized = self.resize_region(cropped_region, (256, 256))

            processed_regions.append(resized)

        return processed_regions

    def crop_image(self, image, region) -> np.ndarray:
        return np.copy(image[region[1]:region[3], region[0], region[2]])

    def resize_region(self, image, new_size: Tuple[int, int]) -> np.ndarray:
        return transform.resize(image, new_size)

    def project_to_latent(self, processed_regions) -> List[ImageColletion]:
        projected = []
        for processed_region in processed_regions:
            projected_w_steps = self.projector.project_image(
                torch.from_numpy(processed_region))
            latents = projected_w_steps[-1].unsqueeze(0)
            synth_image, features = self.projector.get_image_and_features(
                projected_w_steps)

            projected.append(ImageColletion(synth_image, features, latents))

        return projected

    def get_features(self, image_collection: ImageColletion) -> Tuple:
        with torch.no_grad():
            torch.cuda.empty_cache()
            latent = image_collection.latent
            feature = image_collection.feature
            img_gen = image_collection.image
            out = self.few_shot(prediction_utils.concat_features(feature))
            predictions: np.ndarray = out.data.max(1)[1].cpu().numpy()
            # TODO Allow visualisation

            return self.select_regions(predictions)

    def select_regions(self, predictions: np.ndarray) -> Tuple:
        feature_vector = [0] * len(self.classes) -1
        
        for class_id in range(len(self.classes)-1):
            # Convert to 1/0 array
            new_arr = np.copy(predictions)
            new_arr[new_arr!=class_id] = 0
            # Label 
            num_regions = measure.label(new_arr, return_num=True, connectivity=2)
            print(num_regions)
            feature_vector[class_id] = num_regions

        return self._feature_vector(*feature_vector)

    def predict_class(self, feature_vector) -> ClassPrediction:
        return self.svm.get_prediction(feature_vector)

    def filter_predictions(
        self, suggested_regions: List[BoundingBox],
        predictions: List[ClassPrediction]
    ) -> List[Tuple[BoundingBox, ClassPrediction]]:

        def is_class(prediction: ClassPrediction) -> bool:
            return prediction.prediction == self.beast

        zipped = zip(suggested_regions, predictions)
        return list(filter(lambda _, predict: is_class(predict), zipped))


if __name__ == "__main__":
    config_file = Path("./config.yaml")
    
    loaded: Dict[AnyStr, Any] = yaml.load(config_file.read_text())

    predictor_configs: Dict[AnyStr, Any] = loaded.get("beasts", {})

    all_classes = [
        predictor_configs[key]["classes"] for key in predictor_configs
    ]
    print(all_classes)

    predictors = [
        Predictor(beast_name, config["gan_model_path"], config["few_shot"],
                  predictor_configs["svm_model_path"])
        for beast_name, config in predictor_configs.items()
    ]
