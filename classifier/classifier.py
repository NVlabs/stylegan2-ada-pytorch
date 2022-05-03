from typing import AnyStr, List, Tuple, Optional, Dict
from pathlib import Path
import pickle

from sklearn import svm

FeatureVector = Tuple[int]
Label = AnyStr

class Classifier:
    def __init__(self,  feature_descriptors: Dict[AnyStr, int], model_path=Optional[AnyStr]):
        self.feature_descriptors = feature_descriptors

        self._svc = svm.SVC(decision_function_shape='ovr', cache_size=1000)

        if model_path is not None:
            self.load_model(model_path)

    def train(self, labels: List[Label], feature_vectors: List[Tuple[int]]):
        self._svc.fit(labels, feature_vectors)

    def get_prediction(self, feature_vector: FeatureVector) -> AnyStr:
        prediction = self._svc.predict(feature_vector)
        print(prediction)

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

        sort = sorted(feature_vector.items())

        for beast, vector in sort:
            if len(vector) != self.feature_descriptors[beast]:
                raise ValueError(f"Beast '{beast}' has unexpected feature vector length. Got: {len(vector)} Expected: {self.feature_descriptors[beast]}")
            
            feature_vector.extend(vector)

        return tuple(feature_vector)

if __name__=="__main__":
    pass