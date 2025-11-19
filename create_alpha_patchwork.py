"""
Create a patchwork visualization of generated images across different alpha values.

For each alpha value, select n=25 images and arrange them in a 5x5 grid.
Then arrange these grids in a larger grid (e.g., 5x5) to show 25 different alpha values.
This helps visualize colorimetric differences and brightness across alpha values.
"""

import os
import click
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def get_alpha_dirs(base_dir):
    """Get all alpha_* directories and sort them by alpha value."""
    alpha_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith("alpha_"):
            try:
                alpha_value = float(item.replace("alpha_", ""))
                alpha_path = os.path.join(base_dir, item)
                if os.path.isdir(alpha_path):
                    alpha_dirs.append((alpha_value, alpha_path))
            except ValueError:
                continue

    # Sort by alpha value
    alpha_dirs.sort(key=lambda x: x[0])
    return alpha_dirs


def get_images_from_dir(dir_path, max_images=25):
    """Get up to max_images PNG files from a directory."""
    images = []
    for file in sorted(os.listdir(dir_path)):
        if file.endswith(".png"):
            images.append(os.path.join(dir_path, file))
            if len(images) >= max_images:
                break
    return images


def create_grid_from_images(image_paths, grid_size=5):
    """Create a single grid image from a list of image paths."""
    if not image_paths:
        return None

    # Load first image to get dimensions
    first_img = Image.open(image_paths[0])
    img_width, img_height = first_img.size

    # Create a blank canvas for the grid
    grid_img = Image.new(
        "RGB", (img_width * grid_size, img_height * grid_size), color="black"
    )

    # Place images in grid
    for idx, img_path in enumerate(image_paths[: grid_size * grid_size]):
        if idx >= grid_size * grid_size:
            break

        row = idx // grid_size
        col = idx % grid_size

        img = Image.open(img_path)
        grid_img.paste(img, (col * img_width, row * img_height))

    return grid_img


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing alpha_* subdirectories with generated images",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Output path for the patchwork image",
)
@click.option(
    "--images-per-alpha",
    type=int,
    default=25,
    help="Number of images to sample per alpha value (default: 25)",
)
@click.option(
    "--grid-size",
    type=int,
    default=5,
    help="Size of the grid for each alpha patch (default: 5x5)",
)
@click.option(
    "--max-alphas",
    type=int,
    default=25,
    help="Maximum number of alpha values to include (default: 25)",
)
@click.option(
    "--patchwork-cols",
    type=int,
    default=5,
    help="Number of columns in the main patchwork grid (default: 5)",
)
def create_patchwork(
    input_dir, output_path, images_per_alpha, grid_size, max_alphas, patchwork_cols
):
    """Create a patchwork visualization of images across different alpha values."""

    print(f"Scanning directory: {input_dir}")
    alpha_dirs = get_alpha_dirs(input_dir)

    if not alpha_dirs:
        print(f"No alpha_* directories found in {input_dir}")
        return

    print(f"Found {len(alpha_dirs)} alpha directories")

    # Limit to max_alphas
    alpha_dirs = alpha_dirs[:max_alphas]
    print(f"Using {len(alpha_dirs)} alpha values")

    # Create grids for each alpha
    alpha_grids = []
    alpha_values = []

    for alpha_value, alpha_path in alpha_dirs:
        print(f"Processing alpha={alpha_value}...")

        # Get images from this alpha directory
        image_paths = get_images_from_dir(alpha_path, max_images=images_per_alpha)

        if len(image_paths) < images_per_alpha:
            print(
                f"  Warning: Only found {len(image_paths)} images (expected {images_per_alpha})"
            )

        if not image_paths:
            print(f"  Skipping alpha={alpha_value} (no images found)")
            continue

        # Create grid for this alpha
        grid = create_grid_from_images(image_paths, grid_size=grid_size)
        if grid:
            alpha_grids.append(grid)
            alpha_values.append(alpha_value)
            print(
                f"  Created {grid_size}x{grid_size} grid with {len(image_paths)} images"
            )

    if not alpha_grids:
        print("No grids created. Exiting.")
        return

    print(f"\nCreating final patchwork with {len(alpha_grids)} grids...")

    # Calculate patchwork dimensions
    patchwork_rows = (len(alpha_grids) + patchwork_cols - 1) // patchwork_cols

    # Create final patchwork with matplotlib for better control and labels
    fig_width = patchwork_cols * 4
    fig_height = patchwork_rows * 4

    fig, axes = plt.subplots(
        patchwork_rows, patchwork_cols, figsize=(fig_width, fig_height)
    )

    # Handle case where there's only one row or column
    if patchwork_rows == 1 and patchwork_cols == 1:
        axes = np.array([[axes]])
    elif patchwork_rows == 1:
        axes = axes.reshape(1, -1)
    elif patchwork_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each grid
    for idx, (grid, alpha_val) in enumerate(zip(alpha_grids, alpha_values)):
        row = idx // patchwork_cols
        col = idx % patchwork_cols

        ax = axes[row, col]
        ax.imshow(grid)
        ax.set_title(f"α = {alpha_val}", fontsize=14, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for idx in range(len(alpha_grids), patchwork_rows * patchwork_cols):
        row = idx // patchwork_cols
        col = idx % patchwork_cols
        axes[row, col].axis("off")

    plt.suptitle(
        f"Alpha Value Patchwork ({grid_size}×{grid_size} images per alpha)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save the figure
    print(f"Saving patchwork to: {output_path}/alpha_patch_work")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Patchwork created successfully!")
    print(f"  - {len(alpha_grids)} alpha values")
    print(f"  - {grid_size}×{grid_size} images per alpha")
    print(f"  - {patchwork_rows}×{patchwork_cols} patchwork grid")


if __name__ == "__main__":
    create_patchwork()
