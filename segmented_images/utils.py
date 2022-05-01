
from pathlib import Path
from typing import AnyStr, Tuple, List

from PIL import Image
import ruamel.yaml as yaml

BoundingBox = Tuple[int, int, int, int]

def convert_yolo_to_pixels(width: int, height: int, bounding_box: BoundingBox) -> BoundingBox:
    converted = (width * bounding_box[0], height*bounding_box[1], width*bounding_box[2], height*bounding_box[3])
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
    
    for image_id, image_data in data.items():
        if image_data.get("beast") != "pegasus":
            continue
        
        image_file = Path(image_data.get("file-path"))
        bounding_box_file = get_bounding_box_file(image_id, image_file.parent)
        yolo_bounding_boxes = get_bounding_boxes(bounding_box_file)
        if len(yolo_bounding_boxes) == 0:
            print(image_id)
            continue
        
        width, height = Image.open(image_file).size
        converted_bounding_boxes = [convert_yolo_to_pixels(width, height, yolo_bounding_box) for yolo_bounding_box in yolo_bounding_boxes]
        data[image_id]["bounding-boxes"] = converted_bounding_boxes
    
    Path("./dump.yaml").write_text(yaml.dump(data, Dumper=yaml.RoundTripDumper))
