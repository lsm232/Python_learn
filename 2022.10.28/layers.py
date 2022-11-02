import torch
import torch.nn as nn

class FeatureConcat(nn.Module):
    def __init__(self,layers):
        super(FeatureConcat, self).__init__()
        self.layers=layers
        self.multiple=len(layers)
    def forward(self,x,outputs): #outputs里面是记录了routes的输出
        return torch.cat([outputs[i] for i in self.layers],1) if self.multiple else outputs[self.layers[0]]

class WeightedFeatureFusion(nn.Module):
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers=layers
        self.weight=weight
        self.n=len(layers)+1
        if weight:
            self.w=nn.Parameter(torch.zeros(self.n))

    def forward(self,x,outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]
        nx=x.shape[1]
        for i in range(self.n-1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            x+=a
        return x
