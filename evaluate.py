from typing import AnyStr, List, Dict, Tuple, Any
from pathlib import Path
from multiprocessing import Pool, current_process
import time
import random

import ruamel.yaml as yaml
from skimage import io, color
import numpy as np
import PIL

from predict_beast import Predictor, RegionProposal, PartSegmentor

def convert_to_rgb(image: np.array) -> np.array:
            
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    if image.shape[2] == 4: # Image is RGBA
        image = color.rgba2rgb(image)

    return image

def calculate_area(box: Tuple) -> float:
    delta_x = abs(box[2] - box[0])
    delta_y = abs(box[3] - box[1])

    return delta_x * delta_y

def calculate_iou(ground_truth: Tuple, predicted: Tuple) -> float:
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

def region_proposal(image_data: Dict[AnyStr, AnyStr]) -> Tuple[AnyStr, List[Tuple[float, Tuple]]]:
    image = io.imread(str(Path("../mythical-images") / image_data.get("file-path")))
            
    image = convert_to_rgb(image)

    predicted_boxes = RegionProposal(scale=900, sigma=1.2, min_size=6000).get_region_proposals(image)

    ground_truths = image_data.get("bounding-boxes", [])
    iou_values = []

    for box in predicted_boxes:
        for index, ground_truth in enumerate(ground_truths):
            try:
                iou = calculate_iou(ground_truth, box)
            except Exception as e:
                continue
            
            # if iou > iou_values[index][0]:
            iou_values.append((iou, tuple(box)))

    return image_data.get("image-id"), iou_values

def evaluate_region_proposal(images: List[Dict[AnyStr, AnyStr]]) -> Dict[AnyStr, List[Tuple[float, Tuple]]]:
    iou_results = {}
    with Pool(processes=8) as pool:
        iou_results_generator = pool.imap(region_proposal, images, chunksize=1)

        for index, iou_result in enumerate(iou_results_generator):
            iou_results[iou_result[0]] = iou_result[1]

            if index % 50 == 0:
                print("Completed: ", index)

    return iou_results

def part_segmentation(image_data: Dict[AnyStr, AnyStr], part_segmentors: Dict[AnyStr, Any]) -> Tuple[AnyStr, List[List[int]]]:
    # Load in image
    image = PIL.Image.open(str(Path("../mythical-images") / image_data.get("file-path"))).convert('RGB')

    # Crop beast region
    region_suggestions = image_data.get("bounding-boxes", [])
    
    processed_regions = []
    for region in region_suggestions:
        _, height = image.size

        # Reverse Y - Coord for PIL Crop
        region[1] = height - region[1]
        region[3] = height - region[3]
        cropped = image.crop(region)
        processed_regions.append(cropped)
    
    region_vectors = []
    for processed_region in processed_regions:
        feature_vectors = part_segmentors[image_data["beast"]].segment_parts(processed_region)

        region_vectors.append(feature_vectors)

    return tuple([image_data["image-id"], list(zip(tuple(image_data["ground-truth-features"]), region_vectors))])

def evaluate_part_segmentation(images: List[Dict[AnyStr, AnyStr]], segmentor_configs: Dict[AnyStr, Dict[AnyStr, Any]]) -> Dict[AnyStr, Dict[AnyStr, int]]:

    part_segmentors = {}

    for beast, segmentor_config in segmentor_configs.items():
        part_segmentors[beast] = PartSegmentor(**segmentor_config)

    segmentation_results = {} # ImageID: [{beast: feature-vector}]

    for index, image_data in enumerate(images, start=1):
        import time
        start = time.perf_counter()
        part_segmentation_result = part_segmentation(image_data, part_segmentors)
        print(f"Image {index} Took: {(time.perf_counter() - start)} seconds")

        segmentation_results[part_segmentation_result[0]] = part_segmentation_result[1]

        if index % 10 == 0:
            print("Completed: ", index)

    return segmentation_results

def whole_system(image_data: Dict[AnyStr, AnyStr], predictor: Predictor) -> Tuple[AnyStr, List[List[int]]]:
    # Load in image
    image = io.imread(str(Path("segmented_images") / image_data.get("file-path")))

    predictions = predictor.predict_beast(image)

    return tuple([image_data["image-id"], tuple([prediction._asdict() for prediction in predictions])])

def evaluate_whole_system(images: List[Dict[AnyStr, AnyStr]], config) -> Dict[AnyStr, Dict[AnyStr, int]]:
    predictor = Predictor(**config)

    prediction_results = {} # ImageID: [{beast: [Predictions]}]

    for index, image_data in enumerate(images, start=1):
        import time
        start = time.perf_counter()
        predictions = whole_system(image_data, predictor)
        print(f"Image {index} Took: {(time.perf_counter() - start)} seconds")

        prediction_results[predictions[0]] = predictions[1]

        if index % 10 == 0:
            print("Completed: ", index)

    return prediction_results


if __name__=="__main__":
    EVALUATE_REGIONS = False
    EVALUATE_PART_SEGMENTATION = False
    EVALUATE_CLASSIFICATION = False
    EVALUATE_WHOLE_SYSTEM = True
    metadata_file = Path("../mythical-images/image-data.yaml")
    data = yaml.load(metadata_file.read_text(), Loader=yaml.RoundTripLoader)
    config_file_path = Path("./config.yaml")
    
    config_file: Dict[AnyStr, Any] = yaml.load(config_file_path.read_text(), Loader=yaml.RoundTripLoader)

    segmentor_configs = config_file["predictor_config"]["part_segmentor_configs"]

    images = []
    for image_data in data.values():
        # if image_data.get("beast") not in ["harpy"]: #"pegasus", "minotaur"]: # Uncomment to test only certain beasts
        #     continue

        images.append(image_data)

    if EVALUATE_REGIONS:
        start = time.perf_counter()
        results = evaluate_region_proposal(images)
        print(f"Took: {(time.perf_counter() - start)/60} minutes")
        # print(results)
        for image_key, result in results.items():
            data[image_key]["iou"] = result

        Path(f"./evaluate_region_proposal_all_6000.yaml").write_text(yaml.dump(data, Dumper=yaml.RoundTripDumper))

    if EVALUATE_PART_SEGMENTATION:
        start = time.perf_counter()
        sample_per_class = 30
        image_by_beast = {}
        for image_data in images:
            curr = image_by_beast.get(image_data["beast"], [])
            curr.append(image_data)
            image_by_beast[image_data["beast"]] = curr

        sample = []
        for all_imgs in image_by_beast.values():
            sample.extend(random.sample(all_imgs, sample_per_class))

        results = evaluate_part_segmentation(sample, segmentor_configs)
        print(f"Evaluation Took: {(time.perf_counter() - start)/60} minutes")
        for image_key, feature_vectors in results.items():
            data[image_key]["feature-vectors"] = feature_vectors

        Path("./evaluate_part_segmentation.yaml").write_text(yaml.dump(data, Dumper=yaml.RoundTripDumper))

    if EVALUATE_WHOLE_SYSTEM:
        start = time.perf_counter()
        sample_per_class = 10
        image_by_beast = {}
        for image_data in images:
            curr = image_by_beast.get(image_data["beast"], [])
            curr.append(image_data)
            image_by_beast[image_data["beast"]] = curr

        sample = []
        for all_imgs in image_by_beast.values():
            sample.extend(random.sample(all_imgs, sample_per_class))

        results = evaluate_whole_system(sample, config_file["predictor_config"])
        print(results)
        print(f"Evaluation Took: {(time.perf_counter() - start)/60} minutes")
        for image_key, feature_vectors in results.items():
            data[image_key]["prediction"] = feature_vectors

        Path("./evaluate_whole_system.yaml").write_text(yaml.dump(data, Dumper=yaml.RoundTripDumper))