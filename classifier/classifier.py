import random
from typing import AnyStr, List, Tuple, Optional, Dict
from pathlib import Path
import pickle
import copy

import numpy as np
from sklearn import svm
import ruamel.yaml as yaml

FeatureVector = Tuple[int]
Label = AnyStr

class Classifier:
    def __init__(self,  feature_descriptors: Dict[AnyStr, int], model_path: Optional[AnyStr] = None):
        self.feature_descriptors = feature_descriptors
        self.classes = sorted(feature_descriptors.keys())

        self._svc = svm.SVC(decision_function_shape='ovr', cache_size=1000, probability=True)

        if model_path is not None:
            print(model_path)
            self.load_model(Path(model_path))

    def train(self, labels: List[Label], feature_vectors: List[Tuple[int]]):
        self._svc.fit(labels, feature_vectors)

    def get_prediction(self, feature_vector: FeatureVector) -> AnyStr:
        prediction = self._svc.predict_proba([feature_vector])
        max_prediction = np.max(prediction[0])
        if max_prediction > 0.9:
            return self.classes[np.argmax(prediction[0])]

        return None

    def save_model(self, file_path: Path):
        if file_path.exists():
            print("Overwriting existing file")

        file_path.write_bytes(pickle.dumps(self._svc))

    def load_model(self, file_path: Path):
        if not file_path.exists():
            raise ValueError(f"File Path: {file_path} does not exist")

        self._svc = pickle.loads(file_path.read_bytes())

    def construct_feature_vector(self, feature_vectors: Dict[AnyStr, FeatureVector]) -> Tuple:
        if len(feature_vectors) != len(self.feature_descriptors):
            raise ValueError(f"Incorrect number of feature vectors given got '{len(feature_vectors)}' expected '{len(self.feature_descriptors)}'")

        feature_vector = []

        sort = sorted(feature_vectors.items())

        for beast, vector in sort:
            if len(vector) != self.feature_descriptors[beast]:
                raise ValueError(f"Beast '{beast}' has unexpected feature vector length. Got: {len(vector)} Expected: {self.feature_descriptors[beast]}")
            
            feature_vector.extend(vector)

        return tuple(feature_vector)

if __name__=="__main__":
    TRAINING_RANGE = 0
    image_data_path = Path("../segmented_images/image-data.yaml")
    config_path = Path("../config.yaml")

    config = yaml.load(config_path.read_text(), Loader=yaml.RoundTripLoader)
    image_data_collection = yaml.load(image_data_path.read_text(), Loader=yaml.RoundTripLoader)

    beast_feature_vectors = {}
    feature_descriptors = {}
    
    for beast, predictor_config in config["predictor_config"]["part_segmentor_configs"].items():
        feature_descriptors[beast] = len(predictor_config["classes"])

    for image_data in image_data_collection.values():
        current = beast_feature_vectors.get(image_data["beast"], [])

        current.extend(image_data["ground-truth-features"])
        beast_feature_vectors[image_data["beast"]] = current

    classifier = Classifier(feature_descriptors)

    # Create Training Dataset
    empty_feature_vectors = {beast: [0] * num_classes for beast, num_classes in feature_descriptors.items()}

    complete_train_set = [] # (feature_vector, beast)
    complete_test_set = []
    for beast, feature_descriptor in beast_feature_vectors.items():
        beast_data_set = []
        for feature_vector in feature_descriptor:
            new_feature_vector = copy.deepcopy(empty_feature_vectors)
            for k, v in new_feature_vector.items():
                new_feature_vector[k] = [i -(TRAINING_RANGE/2) + (TRAINING_RANGE*random.random()) for i in v]
            new_feature_vector[beast] = feature_vector
            # for k, v in new_feature_vector.items():
            #     new_feature_vector[k] = [i -(TRAINING_RANGE/2) + (TRAINING_RANGE*random.random()) for i in v]
            beast_data_set.append((classifier.construct_feature_vector(new_feature_vector), beast))

        num_train = int(0.8 * len(beast_data_set))
        print(f"Len Beast: {len(beast_data_set)} Num Train: {num_train}")
        beast_train_set = []
        random.shuffle(beast_data_set)
        for i in range(num_train):
            beast_train_set.append(beast_data_set.pop(0))


        complete_train_set.extend(beast_train_set)
        complete_test_set.extend(beast_data_set)


    classifier.train(*zip(*complete_train_set))

    confusion_matrix = {
        beast_name: {beast_name_inner: 0 for beast_name_inner in feature_descriptors.keys()}
        for beast_name in feature_descriptors.keys()
    }

    for vector, beast in complete_test_set:
        prediction = classifier.get_prediction(vector)
        print(prediction)
        confusion_matrix[beast][prediction] += 1

    print(confusion_matrix)

    classifier.save_model(Path("../trained/classifier.pkl"))
