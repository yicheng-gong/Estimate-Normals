import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return map(self.lambda_func,self.forward_prepare(input))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))
    
class BasicBlock(nn. Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        identity = self.shortcut(identity)

        x += identity
        x = self.relu(x)

        return x

def make_layer(out_channels, num_blocks, stride):
    in_channels = 50
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(BasicBlock(in_channels, out_channels, stride))
        in_channels = out_channels * BasicBlock.expansion
        
    return nn.Sequential(*layers)

def create_model():
    model = nn.Sequential(
        nn.Conv2d(3, 50, kernel_size=3),
        nn.BatchNorm2d(50),
        nn.ReLU(inplace=True),
        make_layer(50, 2, stride=1),
        make_layer(50, 2, stride=2),
        make_layer(50, 2, stride=2),
        make_layer(50, 2, stride=2),
        Lambda(lambda x: x.view(x.size(0),-1)), # View,
        nn.Sequential( # Sequential,
    		nn.Dropout(0.5),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(800,256)), # Linear,
    		nn.ReLU(),
    		nn.Dropout(0.5),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(256,128)), # Linear,
    		nn.ReLU(),
    		nn.Dropout(0.5),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(128,64)), # Linear,
    		nn.ReLU(),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(64,2)), # Linear,
    	),
    )
    return model

def load_model(filename):
    model = create_model()
    model.load_state_dict(torch.load(filename))
    return model