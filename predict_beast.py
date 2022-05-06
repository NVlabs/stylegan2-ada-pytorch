from dataclasses import dataclass
from typing import AnyStr, List, Tuple, Dict, Any, Optional
from pathlib import Path
from collections import namedtuple
from skimage import io, color
import numpy as np
import PIL

import ruamel.yaml as yaml

from region_proposal.region_proposal import RegionProposal
from part_segmentation.part_segmentor import PartSegmentor
from classifier.classifier import Classifier

Config = Dict[AnyStr, Any]

ClassPrediction = namedtuple('ClassPrediction', ['prediction', 'region_id'])
       
class Predictor:
    def __init__(self,
            region_proposal_config: Optional[Config] = None, 
            part_segmentor_configs: Optional[Dict[AnyStr, Config]] = None, 
            classifier_config: Optional[Config] = None,
        ) -> None:

        # Use default config
        if region_proposal_config is None:
            region_proposal_config = {}

        if part_segmentor_configs is None or len(part_segmentor_configs) == 0:
            raise ValueError("Invalid PartSegmentor configs")

        if classifier_config is None:
            classifier_config = {}

        self.classes = [beast_name for beast_name in part_segmentor_configs]

        self.region_proposal = RegionProposal(**region_proposal_config)
        
        self.part_segmentors = {}
        feature_descriptors = {}
        for beast_name, part_segmentor_config in part_segmentor_configs.items():
        
            self.part_segmentors[beast_name] = PartSegmentor(**part_segmentor_config)
            feature_descriptors[beast_name] = len(part_segmentor_config["classes"])

        self.classifier: Classifier = Classifier(feature_descriptors, **classifier_config)

    def predict_beast(self, image: np.ndarray) -> List[ClassPrediction]:
        image = np.copy(image)
        if image.shape[2] == 2: # Convert Gray -> RGB
            image = color.gray2rgb(image)

        if image.shape[2] == 4: # Convert RBGA -> RGB
            image = color.rgba2rgb(image)
            image = image * 255
            image = image.astype(np.uint8)


        suggested_regions = self.region_proposal.get_region_proposals(image)
        processed_regions: List[PIL.Image] = self.process_regions(suggested_regions, image)

        feature_vectors = []
        for processed_region in processed_regions:
            features: Dict[AnyStr, Tuple] = {}
            for beast, part_segmentor in self.part_segmentors.items():
                features[beast] = part_segmentor.segment_parts(processed_region)

            feature_vectors.append(self.classifier.construct_feature_vector(features))

        predicted_classes = []
        for feature_vector in feature_vectors:
            prediction = self.classifier.get_prediction(feature_vector)
            if prediction is not None:
                predicted_classes.append(prediction)

        results = self.filter_predictions(suggested_regions, predicted_classes)

        return results

    def process_regions(self, region_suggestions: Tuple, image: np.array) -> List[PIL.Image.Image]:
        image = PIL.Image.fromarray(image)
        processed_regions = []
        for suggested_region in region_suggestions:
            cropped_region = self.crop_image(image, suggested_region)
            processed_regions.append(cropped_region)

        return processed_regions

    def crop_image(self, image: PIL.Image, region) -> PIL.Image:
        _, height = image.size

        # Reverse Y - Coord for PIL Crop
        new = list(region)
        new[1] = height - region[1]
        new[3] = height - region[3]
        cropped = image.crop(new)
        return cropped

    def filter_predictions(self, suggested_regions: List[Tuple], predictions: List[AnyStr]) -> List[ClassPrediction]:
        def is_class(prediction: ClassPrediction) -> bool:
            return prediction.prediction in self.classes

        regions_with_predictions = [ClassPrediction(prediction, suggested_region) for prediction, suggested_region in zip(predictions, suggested_regions)]
        return list(filter(is_class, regions_with_predictions))

if __name__ == "__main__":
    config_file_path = Path("./config.yaml")
    
    config_file: Dict[AnyStr, Any] = yaml.load(config_file_path.read_text())

    predictor = Predictor(**config_file["predictor_config"])

    test_images_path = Path(config_file["images_path"])

    image_data = yaml.load(test_images_path.read_text(), Loader=yaml.RoundTripLoader)
    
    images = []
    for image_data in image_data.values():
        if config_file.get("test_beasts") is not None and image_data.get("beast") not in config_file.get("test-beasts"):
            continue

        images.append(image_data)

    for image in images:
        image = io.imread(image['file-path'], mode="RGB")
        predictions = predictor.predict_beast()

