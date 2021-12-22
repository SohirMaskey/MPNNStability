"""
In this file we build the models and save them for later puposes.
We use the model classes to contruct models and rebuild the models from its state_dict
"""

import sys
sys.path.insert(1,'../src')
import time


import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


from DataLoader_rad import RGGDataset_rad
from Settings import datapath, num_nodes

"""
DL = RGGDataset_rad(root = datapath)
dataset = DL.get(5, 10)
dataset.x = torch.ones(dataset.num_nodes)
"""

class GCN(torch.nn.Module):
    def __init__(self):
        #super().__init__()
        super(GCN, self).__init__()
        self.lin1 = torch.nn.Linear(1, 4)
        self.lin2 = torch.nn.Linear(4, 1)
        self.graphSage = SAGEConv(1, 1, root_weight=False, bias=False)
        #self.graphSage = SAGEConv(dataset.num_node_features, 1, root_weight=False, bias=False)        
        #self.graphSage = SAGEConv(DL.get(101).num_node_features,1 ,root_weight = False, bias = False)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        #x = F.dropout(x, training=self.training)
        x = self.graphSage(x, edge_index)

        return x
        #F.log_softmax(x, dim=1)

#model = GCN()
        
if __name__ == "__main__":
    model = GCN()
    torch.save(model.state_dict(), '../models/GCNTwoLayersGraphSage')
    
class TwoSageLayers(torch.nn.Module):
    def __init__(self, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = SAGEConv(1, 2*out_channels) # cached only for transductive learning
        self.conv2 = SAGEConv(2*out_channels, 1) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
if __name__ == "__main__":
    model = TwoSageLayers()
    torch.save(model.state_dict(), '../models/GCNTwoLayersGraphSage')
    
    
    
"""
We dont use the continuous MPNN below anymore. Instead we use discrete on the largest graph

"""    
torch.set_printoptions(precision=10) # Set displayed output precision to 10 digits

from torchquad import enable_cuda # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
import torchquad

#helper functions

eucl_norm = lambda x: torch.sqrt(torch.square(x[:,0]) + torch.square(x[:,1]))

def indicator_fct(x, distance, center):
    a = x[:,0] - center[0]
    b = x[:,1] - center[1]
    c = torch.cat((a, b))
    
    d = torch.reshape(c, (2, len(x))).T
    
    z = (eucl_norm(d) < distance)*1
    
    return z

simp = Simpson() 


class cGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(1, 4)
        self.lin2 = torch.nn.Linear(4, 1)
        #self.lin1.load_state_dict(model.lin1.state_dict()) #if I write a new model I need to include all that again
        #self.lin2.load_state_dict(model.lin2.state_dict())
        
        self.graphSage = SAGEConv(1, 1 ,root_weight=False, bias=False)
        #self.graphSage.load_state_dict(model.graphSage.state_dict())
        
        #self.weight = self.graphSage.lin_l.weight
        
    def forward(self, input_fct, radius=0.5):

        self.input_fct = input_fct
        
        def output_fct(x):
            y = self.input_fct(x)
            y = torch.reshape(y, (len(y), 1))
            z = self.lin1(y)
            return self.lin1(y) 
        
        def output_fct2(x):
            return F.relu(output_fct(x))
        
        def output_fct3(x):
            y = self.lin2(output_fct2(x))
            return y.flatten()
        
        """
        def output_fct(x):
            start = time.time()

            y = input_fct(x)
            b = torch.empty(( 0 ))
            for i, w in enumerate(y):
                b = torch.cat((b, self.lin1(torch.tensor([w]))), 0)
            d = torch.reshape(b,(len(x),4))
            
            end = time.time()
            print(f"1st layer cGCN: Took {(end-start)* 1000.0:.3f} ms")
            return d

        def output_fct2(x):
            return F.relu(output_fct(x))
        
        def output_fct3(x):
            start = time.time()
            
            y = output_fct(x)
            b = torch.empty(( 0 ))
            for i, w in enumerate(y):
                b = torch.cat((b, self.lin2(torch.tensor(w))), 0)
            #d = torch.reshape(b,(4, len(x)))
            
            end = time.time()
            print(f"2nd layer cGCN: Took {(end-start)* 1000.0:.3f} ms")
            return b
        """
        
        def output_fct4(position):
            g = lambda x, distance, center: indicator_fct(x, distance, position)*output_fct3(x)
            
            def h(x):
                y = g(x, radius, position)
                return y
            #h = lambda x: g(x, radius, position)
            l = lambda x: indicator_fct(x, radius, position)
            
            int_of_f = simp.integrate(h, dim=2, N=100001, integration_domain = [[0,1], [0,1]])
            measure_Ball = simp.integrate(l, dim=2, N=100001, integration_domain = [[0,1], [0,1]])
        
            return torch.tensor(self.graphSage.lin_l.weight*int_of_f/measure_Ball)
        
        return output_fct4
    #F.log_softmax(x, dim=1)
    
if __name__ == "__main__":
    cGCNN = cGCN()
    torch.save(cGCNN.state_dict(), '../models/cGCNTwoLayersGraphSage')