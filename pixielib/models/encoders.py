import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class ResnetEncoder(nn.Module):
    def __init__(self, append_layers = None, device = 'cpu'):
        super(ResnetEncoder, self).__init__()
        from . import resnet
        # feature_size = 2048
        self.feature_dim = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        # regressor
        self.append_layers = append_layers
        # for normalize input images
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        # Register MEAN and STD as buffers and move them to CUDA
        self.register_buffer('MEAN', torch.tensor(MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer('STD', torch.tensor(STD).view(1, 3, 1, 1).to(device))

        
    def forward(self, inputs):
        ''' inputs: [bz, 3, h, w], range: [0,1]
        '''
        inputs = (inputs - self.MEAN)/self.STD
        features = self.encoder(inputs)
        if self.append_layers:
            features = self.last_op(features)
        return features

class MLP(nn.Module):
    def __init__(self, channels = [2048, 1024, 1], last_op = None):
        super(MLP, self).__init__()
        layers = []

        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l+1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        if last_op:
            layers.append(last_op)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outs = self.layers(inputs)
        return outs

class HRNEncoder(nn.Module):
    def __init__(self, append_layers = None, device = 'cpu'):
        super(HRNEncoder, self).__init__()
        from . import hrnet
        self.feature_dim = 2048
        self.encoder = hrnet.load_HRNet(pretrained=True) #out: 2048
        # regressor
        self.append_layers = append_layers
        # for normalize input images
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        # Register MEAN and STD as buffers and move them to CUDA
        self.register_buffer('MEAN', torch.tensor(MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer('STD', torch.tensor(STD).view(1, 3, 1, 1).to(device))

        
    def forward(self, inputs):
        ''' inputs: [bz, 3, h, w], range: [0,1]
        '''
        ## print the device of the inputs
        print('inputs.device:', inputs.device)

        inputs = (inputs - self.MEAN)/self.STD ## correct this, rewrite the normalization
        


        features = self.encoder(inputs)['concat']
        if self.append_layers:
            features = self.last_op(features)
        return features   