from typing import List, Tuple
import torch

@torch.no_grad()
def concat_features(features):
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)

class FeatureVectorConstructor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is  None:
            cls._instance = super.__new__(cls, *args, **kwargs)
        
        return cls._instance

    def __init__(self, classes, feature_vectors) -> None:
        self.class_ids = {class_name: index for index, class_name in enumerate(classes)}
        self.feature_vector_length = sum(len(feature_vector for feature_vector in feature_vectors))

        self.offsets: List[int] = [0, *[len(feature_vector) for feature_vector in feature_vectors]]
        self.offsets.pop()

    def construct_feature_vector(self, class_name, feature_vector) -> Tuple[int]:
        class_id = self.class_ids[class_name]
        offset = self.offsets[class_id]

        new_feature_vector = [0] * self.feature_vector_length

        for index, value in enumerate(feature_vector):
            new_feature_vector[index+offset] = value
        
        return new_feature_vector