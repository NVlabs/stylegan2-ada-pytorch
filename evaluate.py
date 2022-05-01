from typing import AnyStr, List, Dict, Tuple, Any
from pathlib import Path
import functools
from multiprocessing import Pool, current_process
from unittest import result

import ruamel.yaml as yaml
import selectivesearch
from skimage import io, color
import numpy as np

from predict_beast import BoundingBox, Predictor

def convert_to_rgb(image: np.array) -> np.array:
            
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    if image.shape[2] == 4: # Image is RGBA
        image = color.rgba2rgb(image)

    return image

def calculate_area(box: BoundingBox) -> float:
    delta_x = abs(box[2] - box[0])
    delta_y = abs(box[3] - box[1])

    return delta_x * delta_y

def calculate_iou(ground_truth: BoundingBox, predicted: BoundingBox) -> float:
    area_gt = calculate_area(ground_truth)
    area_pr = calculate_area(predicted)

    overlap_box = tuple([
        max(ground_truth[0], predicted[0]),
        max(ground_truth[1], predicted[1]),
        min(ground_truth[2], predicted[2]),
        min(ground_truth[3], predicted[3]),
    ])


    area_overlap = calculate_area(overlap_box)


    total_area = area_gt + area_pr - area_overlap
    
    iou = area_overlap / total_area

    if iou < 0 or iou > 1:
        raise Exception(f"Invalid IoU")

    return iou

def convert_to_boxes(predicted_boxes) -> List[BoundingBox]:
    new_boxes = []
    for predicted_box in predicted_boxes:
        # (minx, miny, width, height)
        top_left = (predicted_box[0], predicted_box[1] +  predicted_box[3])
        bottom_right = (predicted_box[0] + predicted_box[2], predicted_box[1])
        new_boxes.append(tuple([*top_left, *bottom_right]))
    return new_boxes

def region_proposal(image_data: Dict[AnyStr, AnyStr]) -> Tuple[AnyStr, List[Tuple[float, BoundingBox]]]:
    image = io.imread(str(Path("segmented_images") / image_data.get("file-path")))
            
    image = convert_to_rgb(image)
    
    _, results = selectivesearch.selective_search(image, scale=900, sigma=1.2, min_size=300)
    predicted_boxes = [tuple(result["rect"]) for result in results]
    predicted_boxes = convert_to_boxes(predicted_boxes)

    ground_truths = image_data.get("bounding-boxes", [])
    iou_values = [(0, None)] * len(ground_truths)

    for box in predicted_boxes:
        for index, ground_truth in enumerate(ground_truths):
            try:
                iou = calculate_iou(ground_truth, box)
            except Exception as e:
                continue
            
            if iou > iou_values[index][0]:
                iou_values[index] = (iou, box)

    return image_data.get("image-id"), iou_values

def evaluate_region_proposal(images: List[Dict[AnyStr, AnyStr]]) -> Dict[AnyStr, List[Tuple[float, BoundingBox]]]:
    iou_results = {}
    with Pool(processes=16) as pool:
        iou_results_generator = pool.imap_unordered(region_proposal, images, chunksize=16)

        for index, iou_result in enumerate(iou_results_generator):
            iou_results[iou_result[0]] = iou_result[1]

            if index % 50 == 0:
                print("Completed: ", index)

    return iou_results

def part_segmentation_map_fun(image_data: Dict[AnyStr, AnyStr], predictor: Predictor) -> Tuple[AnyStr, List[int]]:
    return part_segmentation(image_data, predictor)

def part_segmentation(image_data: Dict[AnyStr, AnyStr], predictor: Predictor) -> Tuple[AnyStr, List[List[int]]]:
    # Load in image
    image = io.imread(str(Path("segmented_images") / image_data.get("file-path")))
            
    image = convert_to_rgb(image)
    
    # Crop beast region
    region_suggestions = image_data.get("bounding-boxes", [])
    
    processed_regions = predictor.process_regions(region_suggestions, image)

    projected_images = predictor.project_to_latent(processed_regions)

    features: List[predictor._feature_vector] = []
    for projected_image in projected_images:
        features.append(predictor.get_features(projected_image))

    print(features)
    return tuple(image_data.get("image-id"), features)

def evaluate_part_segmentation(images: List[Dict[AnyStr, AnyStr]]) -> Dict[AnyStr, Dict[AnyStr, int]]:

    config_file = Path("./config.yaml")
    
    loaded: Dict[AnyStr, Any] = yaml.load(config_file.read_text())

    predictor_configs: Dict[AnyStr, Any] = loaded.get("beasts", {})

    predictor_config = predictor_configs["pegasus"]

    segmentation_results = {}

    with Pool(processes=4) as pool:
        predictor = Predictor("pegasus", predictor_config["gan_model_path"], predictor_config["few_shot"], predictor_configs["svm_model_path"])
        part_result_generation = pool.imap_unordered(functools.partial(part_segmentation_map_fun, image_data, predictor), images, chunksize=2)

        for index, part_segmentation_result in enumerate(part_result_generation):
            segmentation_results[part_segmentation_result[0]] = part_segmentation_result[1]

            if index % 50 == 0:
                print("Completed: ", index)

    return segmentation_results

if __name__=="__main__":
    metadata_file = Path("./segmented_images/dump.yaml")
    data = yaml.load(metadata_file.read_text(), Loader=yaml.RoundTripLoader)
    
    images = []
    for image_data in data.values():
        if image_data.get("beast") != "pegasus":
            continue

        images.append(image_data)

    import time
    start = time.perf_counter()
    results = evaluate_region_proposal(images)
    print(f"Took: {(time.perf_counter() - start)/60} minutes")
    for image_key, best_iou in results.items():
        data[image_key]["iou"] = best_iou

    Path("./evaluate_region_proposal.yaml").write_text(yaml.dump(data, Dumper=yaml.RoundTripDumper))