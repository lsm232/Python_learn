import torch
import torch.nn as nn

class FeatureConcat(nn.Module):
    def __init__(self,layers):
        super(FeatureConcat, self).__init__()
        self.layers=layers
        self.multiple=len(layers)>1
    def forward(self,outputs):
        return torch.cat([outputs[i] for i in self.layers] if self.multiple else outputs[self.layers[0]])

class WeightedFeatureFusion(nn.Module):
    def __init__(self,layers):
        super(WeightedFeatureFusion, self).__init__()
        self.layers=layers
        self.n=len(self.layers)
    def forward(self,outputs,x):
        for i in range(self.n):
            x=x+outputs[self.layers[i]]
        return x


