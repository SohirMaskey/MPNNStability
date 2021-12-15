"""
I want to add some helper functions
"""

import numpy as np
import torch
from torch_geometric.data import Data
import time

from torch_geometric.nn import SAGEConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

import sys
sys.path.insert(1,'../src')
from DataLoader_rad import RGGDataset_rad

#Visualization

from torch_geometric.utils import to_networkx

import networkx as nx


def draw_graph(graph, color, pos):
    """
    input the graph as networkx graph, the positions of the nodes as a dictionary and the output of the mode
    """
    #to write with the things below.
"""
DL = RGGDataset_grid(root = '../../input_rad')

dataset = DL.get(1,2**5)

positions = torch.load('../../input_rad/raw/grid_positions_128.pt')

model = GCN()
model.load_state_dict(torch.load( '../models/GCNTwoLayersGraphSage'))

data = DL.get(2, 2**8) 
pos = positions[7]

hi = to_networkx(data, to_undirected = True)

pos2 = {i: pos[0][i].detach().numpy() for i in range(256)}

signal2 = signal[pos[1].type(torch.LongTensor)]
signal2 = torch.reshape(signal2,( len(signal2),1))
data.x = signal2 # + (0.01**0.5)*torch.randn(len(signal),1) #random noise

output = model.forward(data).detach().numpy().flatten()

nx.draw(hi, node_size=30, node_color=output, width=.1, node_shape='.',
            edge_color='gray', pos = pos2)
plt.plot()
plt.savefig('../output/graph.png', dpi=600) #for higher quality
"""
    
#Helper for creating signal, e.g., band-limited Fourier Function, it must read in the largest position and output a value on each pos


"""
Don't know why, but does not work
"""
def create_randn_bandlimited(gridsize=256, sup=20, ampl=3):
    #gridsize=int(gridsize)
    #sup=int(sup) 
    #ampl=int(ampl)
    support = torch.randn(sup, sup, dtype=torch.complex64) #we create the support of the signal in Fourier domain
    square = torch.zeros(gridsize,gridsize,dtype = torch.complex64)
    square[:sup,-sup:] = support
    square[:ampl,-ampl:]  =  square[:ampl,-ampl:]*100
    ifft2 = torch.fft.ifft2(square) #now we have the signal in time domain 
    ifft2_real = ifft2.real
    return ifft2_real

def fct_eval(csignal, tensor): 
    """
    For evaluating the signal we got above in a vector in [0,1]^2. The idea is that the IFFT gives us a 256x256
    grid. For every 2D vector, we check by the above calculations in which grid-patch it lies and then
    evealuate the signal we receive by IFFT in it.
    """
    
    i = tensor[0][:,0]*257-1
    i = i.type(torch.LongTensor) 
    j = tensor[0][:,1]*257-1
    j = j.type(torch.LongTensor)
    return csignal[i,j]
    
#Error Calculation
    
    
if __name__ == "__main__":
    pass
