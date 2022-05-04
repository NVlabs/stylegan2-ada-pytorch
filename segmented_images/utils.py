
from pathlib import Path
from typing import AnyStr, Tuple, List

from PIL import Image
import ruamel.yaml as yaml

BoundingBox = Tuple[float, float, float, float]

# yolo structure center_x, center_y, width, height
def convert_yolo_to_pixels(width: int, height: int, bounding_box: BoundingBox) -> BoundingBox:
    center_x = width * bounding_box[0]
    center_y = height * bounding_box[1]
    half_width = 0.5 * width * bounding_box[2]
    half_height = 0.5 * height * bounding_box[3]
    top_left_x = center_x - half_width
    top_left_y = center_y + half_height
    bottom_right_x = center_x + width
    bottom_right_y = top_left_y - half_height
    converted = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    return tuple(int(item) for item in converted)

def get_bounding_boxes(file_path: Path) -> List[BoundingBox]:
    file_contents = file_path.read_text()
    values = file_contents.split("\n")
    values.pop(-1)

    bounding_boxes = []
    for value in values:
        value = value.split(' ')
        value.pop(0) # Remove class label
        bounding_box = tuple(float(i) for i in value)
        bounding_boxes.append(bounding_box)

    return bounding_boxes

def get_bounding_box_file(image_id: AnyStr, base_path: Path) -> Path:
    bounding_box_path = base_path / f"{image_id}.txt"
    return bounding_box_path

if __name__=="__main__":
    metadata_file = Path("./image-data.yaml")
    data = yaml.load(metadata_file.read_text(), Loader=yaml.RoundTripLoader)
    
    to_remove = []
    for image_id, image_data in data.items():
        # if image_data.get("beast") != "harpy":
        #     continue
        
        image_file = Path(image_data.get("file-path"))
        bounding_box_file = get_bounding_box_file(image_id, image_file.parent)
        if not bounding_box_file.exists():
            # to_remove.append(image_id)
            print(image_id)
            continue

        yolo_bounding_boxes = get_bounding_boxes(bounding_box_file)
        if len(yolo_bounding_boxes) == 0:
            print(image_id)
            # to_remove.append(image_id)
            continue
        
        width, height = Image.open(image_file).size
        converted_bounding_boxes = [convert_yolo_to_pixels(width, height, yolo_bounding_box) for yolo_bounding_box in yolo_bounding_boxes]
        data[image_id]["bounding-boxes"] = converted_bounding_boxes
    

    for image_id in to_remove:
        del data[image_id]
        try:
            image_file.unlink()
        except FileNotFoundError:
            pass

    Path("./dump.yaml").write_text(yaml.dump(data, Dumper=yaml.RoundTripDumper))
