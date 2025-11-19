# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import warnings
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import matplotlib.pyplot as plt

import legacy
from age_estimator import AgePredictor

# Suppress CUDA kernel compilation warnings
warnings.filterwarnings('ignore', message='Failed to build CUDA kernels for upfirdn2d')

# ----------------------------------------------------------------------------


def float_range(s: str) -> List[float]:

    parts = s.split(":")
    if len(parts) == 3:
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        return list(np.round(np.linspace(start, stop, num), decimals=1))
    else:
        raise ValueError(
            f"Linspace format requires exactly 3 values (start:stop:num), got {len(parts)}"
        )
    
def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option(
    "--vgg-path",
    help="Path to vgg age predictor",
    type=str,
    metavar="FILE",
)
@click.option("--seeds", type=num_range, help="List of random seeds")
@click.option(
    "--trunc",
    "truncation_psi",
    type=float,
    help="Truncation psi",
    default=1,
    show_default=True,
)
@click.option(
    "--class",
    "class_idx",
    type=int,
    help="Class label (unconditional if not specified)",
)
@click.option(
    "--noise-mode",
    help="Noise mode",
    type=click.Choice(["const", "random", "none"]),
    default="const",
    show_default=True,
)
@click.option("--projected-w", help="Projection result file", type=str, metavar="FILE")
@click.option(
    "--outdir",
    help="Where to save the output images",
    type=str,
    required=True,
    metavar="DIR",
)
@click.option(
    "--weight-vector",
    help="Path to weight vector file (.npy)",
    type=str,
    metavar="FILE",
)

@click.option(
    "--alphas",
    type=float_range,
    help='Alpha values for weight modulation (e.g., "-10,0,10" or "-10-10")',
    default="-5, -2, -1, 0, 1, 2, 5",
    show_default=True,
)
@click.option(
    "--style-range",
    type=(int, int),
    help="Range of style blocks to apply weight vector (start, end). Coarse: 0-8, Middle: 4-12, Fine: 12-17. Default: all (0, 17)",
    default=(0, 17),
    show_default=True,
)
@click.option(
    "--create-composite",
    type=bool,
    help="Bolean to create or not a composite grid of the generated images",
    default=False,
    show_default=True,
)

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    vgg_path: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    weight_vector: Optional[str],
    alphas: List[int],
    style_range: tuple,
    create_composite: bool = False,
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
    
    \b
    # (Tim) Generate FFHQ images with weight modulation
    python generate.py --outdir=stylegan2-ada-pytorch/out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # (Tim) Generate with weight modulation on coarse styles only (pose, face shape, eyeglasses)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --style-range 0 8 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # (Tim) Generate with weight modulation on middle styles (facial features, hair style, eyes)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --style-range 4 12 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # (Tim) Generate with weight modulation on fine styles (color scheme, microstructure)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --style-range 12 17 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print("warn: --seeds is ignored when using --projected-w")
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)["w"]
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
                f"{outdir}/proj{idx:02d}.png"
            )
        return

    if seeds is None:
        ctx.fail("--seeds option is required when not using --projected-w")

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail(
                "Must specify class label with --class when using a conditional network"
            )
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    # Load weight vector if specified.
    weight_vec = None
    if weight_vector is not None:
        print(f'Loading weight vector from "{weight_vector}"')
        weight_vec = torch.tensor(np.load(weight_vector), device=device)
        print(f"Weight vector shape: {weight_vec.shape}")

        # Validate style range
        start_idx, end_idx = style_range
        if not (0 <= start_idx <= 17 and 0 <= end_idx <= 17):
            ctx.fail("Style range indices must be between 0 and 17")
        if start_idx > end_idx:
            ctx.fail("Style range start must be <= end")
        print(
            f"Applying weight vector to style blocks {start_idx} to {end_idx} (inclusive)"
        )

    # Initialize age predictor if using weight vector
    age_predictor = None
    if weight_vector is not None:
        print("Initializing age predictor...")
        age_predictor = AgePredictor(vgg_path=vgg_path)

    # Generate images.
    all_images = []  # Store images for composite: list of lists (one per seed)

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi)

        if weight_vec is not None:
            start_idx, end_idx = style_range

            # First, check age with alpha=0 if age estimator is available
            if age_predictor is not None:
                img = G.synthesis(w, noise_mode=noise_mode)
                estimated_age = age_predictor(img)

                if estimated_age is None:
                    print(f"  ⚠️  Seed {seed}: No face detected, skipping...")
                    continue
                elif estimated_age <= 20:
                    print(f"  ⚠️  Seed {seed}: Age {estimated_age:.1f} <= 20, skipping...")
                    continue
                else:
                    print(f"  ✓ Seed {seed}: Age {estimated_age:.1f} > 20, proceeding...")

            # If age check passed (or not applicable), generate all alphas
            seed_images = []
            for alpha in alphas:
                # Clone w to avoid modifying the original
                w_modified = w.clone()
                # Apply weight vector only to the specified range of style blocks
                w_modified[:, start_idx : end_idx + 1, :] += alpha * weight_vec[
                    start_idx : end_idx + 1, :
                ].unsqueeze(0)
                assert w_modified.shape[1:] == (G.num_ws, G.w_dim)
                img = G.synthesis(w_modified, noise_mode=noise_mode)
                img = (
                    (img.permute(0, 2, 3, 1) * 127.5 + 128)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                img_array = img[0].cpu().numpy()
                pil_img = PIL.Image.fromarray(img_array, "RGB")

                # Create subdirectory for this alpha value
                alpha_dir = os.path.join(outdir, f"alpha_{alpha}")
                os.makedirs(alpha_dir, exist_ok=True)

                pil_img.save(
                    f"{alpha_dir}/seed{seed:04d}_styles{start_idx}-{end_idx}.png"
                )
                seed_images.append(img_array)
            all_images.append(seed_images)
        else:
            # Generate image without weight modulation
            img = G.synthesis(w, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
                f"{outdir}/seed{seed:04d}.png"
            )

    # Create composite image if weight modulation was used
    if create_composite and weight_vec is not None and len(all_images) > 0:
        print("Creating composite image...")
        num_rows = len(all_images)
        num_cols = len(alphas)
        start_idx, end_idx = style_range

        # Create matplotlib figure
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3)
        )

        # Handle case where there's only one row or one column
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot images in grid
        for row_idx, seed_images in enumerate(all_images):
            for col_idx, img_array in enumerate(seed_images):
                ax = axes[row_idx, col_idx]
                ax.imshow(img_array)
                ax.axis("off")

                # Add column labels (alpha values) on top row
                if row_idx == 0:
                    ax.set_title(f"α={alphas[col_idx]}", fontsize=12)

                # Add row labels (seed numbers) on left column
                if col_idx == 0:
                    ax.set_ylabel(
                        f"Seed {seeds[row_idx]}",
                        fontsize=12,
                        rotation=0,
                        labelpad=40,
                        va="center",
                    )

        # Add overall title
        style_type = "all styles"
        if start_idx <= 8 and end_idx <= 8:
            style_type = "coarse styles (pose, face shape)"
        elif start_idx >= 4 and end_idx <= 12:
            style_type = "middle styles (facial features)"
        elif start_idx >= 12:
            style_type = "fine styles (color, microstructure)"
        fig.suptitle(
            f"Weight Vector Applied to Blocks {start_idx}-{end_idx} ({style_type})",
            fontsize=14,
            y=1.0,
        )

        plt.tight_layout()
        composite_path = f"{outdir}/composite_grid_styles{start_idx}-{end_idx}.png"
        plt.savefig(composite_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Composite image saved to {composite_path}")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    os.chdir("/home/sagemaker-user/stylegan2-ada-pytorch")

    # Default arguments for debugging/development
    if len(sys.argv) == 1:
        sys.argv = [
            "generate.py",
            "--outdir=out",
            "--trunc=0.7",
            "--seeds=600-605",
            "--style-range", "0", "4",
            "--alphas=0:0:1",
            "--weight-vector=weight.npy",
            "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            "--vgg-path=/home/sagemaker-user/stylegan2-ada-pytorch/dex_age_classifier.pth"
        ]

    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
