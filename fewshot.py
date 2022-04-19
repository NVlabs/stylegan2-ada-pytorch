from torch import nn
import torch.nn.functional as F

def ConvLReLU(in_ch, out_ch, kernel_size, dilation):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=dilation, dilation=dilation),
        nn.LeakyReLU(inplace=True)
    )

class FewShotCNN(nn.Module):
    def __init__(self, in_ch, n_class, size='S'):
        super().__init__()
        
        assert size in ['S', 'M', 'L']
        
        dilations = {
            'S': [1, 2, 1, 2, 1],
            'M': [1, 2, 4, 1, 2, 4, 1],
            'L': [1, 2, 4, 8, 1, 2, 4, 8, 1],            
        }[size]
        
        channels = {
            'S': [128, 64, 64, 32],
            'M': [128, 64, 64, 64, 64, 32],
            'L': [128, 64, 64, 64, 64, 64, 64, 32],         
        }[size]
        channels = [in_ch] + channels + [n_class]
        
        layers = []
        for d, c_in, c_out in zip(dilations, channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=d, dilation=d))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers[:-1])
        
    def forward(self, x):
        return self.layers(x)