from PIL import Image
from pathlib import Path

import ruamel.yaml as yaml


def crop_image(image, region):
    _, height = image.size

    # Reverse Y - Coord for PIL Crop
    region[1] = height - region[1]
    region[3] = height - region[3]
    cropped = image.crop(region)
    return cropped

image_datas = yaml.load(Path("./dump.yaml").read_text(), Loader=yaml.RoundTripLoader)

count = 0
outpath = Path("./harpy_gan")
for image_data in image_datas.values():
    if image_data.get("beast") != "harpy":
        continue

    image = Image.open(str(image_data.get("file-path"))).convert('RGB')

    cropped_regions = [crop_image(image, region) for region in image_data.get("bounding-boxes")]

    for cropped_region in cropped_regions:
        count += 1
        cropped_region.save(str(outpath/Path(f"{count}.png")), "PNG")



    