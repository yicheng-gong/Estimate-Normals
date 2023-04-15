# Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
# Copyright (c) 2016 Alexande Boulch and Renaud Marlet
#
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
# Fifth Floor, Boston, MA 02110-1301  USA
#
# PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
# Computer Graphics Forum

import torch
import torch.nn as nn
from torch.autograd import Variable

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

def create_model():
    model= nn.Sequential( # Sequential,
    	nn.Conv2d(5,50,(3, 3)),
    	nn.ReLU(),
    	nn.BatchNorm2d(50),
    	nn.Conv2d(50,50,(3, 3)),
    	nn.ReLU(),
    	nn.BatchNorm2d(50),
    	nn.MaxPool2d((2, 2),(2, 2)),
    	nn.Conv2d(50,96,(3, 3)),
    	nn.ReLU(),
    	nn.MaxPool2d((2, 2),(2, 2)),
    	Lambda(lambda x: x.view(x.size(0),-1)), # View,
    	nn.Sequential( # Sequential,
    		nn.Dropout(0.5),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(3456,2048)), # Linear,
    		nn.ReLU(),
    		nn.Dropout(0.5),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2048,1024)), # Linear,
    		nn.ReLU(),
    		nn.Dropout(0.5),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1024,512)), # Linear,
    		nn.ReLU(),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,2)), # Linear,
    	),
    )
    return model

def load_model(filename):
    model = create_model()
    model.load_state_dict(torch.load(filename))
    return model
