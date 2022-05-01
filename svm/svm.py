from typing import AnyStr, List, Tuple
from pathlib import Path
import pickle

from sklearn import svm

FeatureVector = Tuple[int]
Label = AnyStr

class SVM:
    def __init__(self):
        self._svc = svm.SVC(decision_function_shape='ovr', cache_size=1000)

    def train(self, labels: List[Label], feature_vectors: List[Tuple[int]]):
        self._svc.fit(labels, feature_vectors)

    def get_prediction(self, feature_vector: FeatureVector) -> AnyStr:
        prediction = self._svc.predict(feature_vector)
        print(prediction)

    def save_model(self, file_path: Path):
        if file_path.exists:
            print("Overwriting existing file")

        file_path.write_bytes(pickle.dumps(self._svc))

    def load_model(self, file_path: Path):
        if not file_path.exists:
            raise ValueError(f"File Path: {file_path} does not exist")

        self._svc = pickle.loads(file_path.read_bytes())

if __name__=="__main__":
    pass