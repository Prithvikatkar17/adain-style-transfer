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
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])  # Keep layers up to relu4_1
        enc_layers = list(self.vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])   # relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu4_1

        for names in ["enc_1", "enc_2", "enc_3", "enc_4"]:
            for param in getattr(self, names).parameters():
                param.requires_grad = False



    def forward(self, input , is_test=False):
        h1 = self.enc_1(input)
        h2 = self.enc_2(h1) 
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        if is_test:
            return h4 
        return [h1, h2, h3, h4]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d(1,1,1,1),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self,x):
        return self.decoder(x)