import torch
import torch.nn.functional as F 
from vgg import VGG


class AgePredictor:
    def __init__(self, vgg_path: str):
        self.age_net = VGG()
        ckpt = torch.load(vgg_path, map_location="cpu")['state_dict']
        ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
        self.age_net.load_state_dict(ckpt)
        self.age_net.cuda()
        self.age_net.eval()
        self.min_age = 0
        self.max_age = 100

    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]
        return predict_age

    def extract_age(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # shape: [1, 3, 224, 224]
        predict_age_pb = self.age_net(x)['fc8']
        predicted_age = self.__get_predicted_age(predict_age_pb)
        return predicted_age
    
    def __call__(self, img: torch.tensor) -> float | None:
        """Allow instance to be called directly."""
        return self.extract_age(img)[0]
    