import torch.nn as nn
from torchvision import models

def load_model():

    model = models.vgg16(pretrained=False)

    model.classifier[6] = nn.Linear(4096,2)

    return model

