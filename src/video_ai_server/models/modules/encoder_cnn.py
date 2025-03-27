import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import Inception3

class EncoderCNN(nn.Module):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(EncoderCNN, self).__init__()
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
        
        self.activation = {}

        self.inception.Mixed_7c.register_forward_hook(self.get_activation("Mixed_7c"))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, images):
        _ = self.inception(images)
        
        features = self.activation["Mixed_7c"]
        
        features = self.pool(features)
        
        features = features.view(features.size(0), -1)      

        return features