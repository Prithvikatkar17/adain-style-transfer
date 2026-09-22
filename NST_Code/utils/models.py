import torch.nn as nn 
import torch

class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()


        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),  # Placeholder for the actual VGG layers
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(), # Relu1-1
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(), # Relu1-2
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(), # Relu2-1
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(), # Relu2-2
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(), # Relu3-1
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(), # Relu3-2
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(), # Relu3-3
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(), # Relu3-4
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(), # Relu4-1
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # Relu4-2
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # Relu4-3
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # Relu4-4
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # Relu5-1
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # Relu5-2
            nn.ReflectionPad2d(1,1,1,1),    
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # Relu5-3
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # Relu5-4
        )
        self.vgg.load_state_dict(torch.load('vgg_path'))  # Load pretrained weights
