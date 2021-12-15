"""
We created a dataset with graphs of size 2^1,2^2,..,2^14. This data set is saved in //home/groups/ai/maskey/input_rad/processed. (not anymore I did it locally)
 A graph of size n and constructed with radius r, can be read out by "DL.get(10*r,n)" after constructing the Data loader object, e.g. "DL = RGGDataset_grid(root = '../../input_rad')".

Then, we use the finest graph(size 2^16) as the continuous limit object and calculate graph wise l^2 errors with some graph signal. 

"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import time

import random
from torch_geometric.utils import from_networkx

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from torch_geometric.nn import SAGEConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

import sys
sys.path.insert(1,'../src')
from DataLoader_rad_grid import RGGDataset_grid
from TwoLayerGraphSage import GCN, cGCN
from Utils import *

import os.path as osp

from torch_geometric.data import Dataset, download_url


import os

DL = RGGDataset_grid(root = '../../input_rad')

dataset = DL.get(1,2**5)

#load the positions, list of positions for all graphs
positions = torch.load('../../input_rad/raw/grid_positions_128.pt')

model = GCN()
model.load_state_dict(torch.load( '../models/GCNTwoLayersGraphSage'))

N = 14

def error_fct(radius, signal):
    L2Errors = []

    cdata = DL.get(radius,2**N)
    cpos = positions[-1]

    cdata.x = signal 

    output = model.forward(cdata)

        
    for i in range(1, N):
        start = time.time()
        data = DL.get(radius, 2**i) 
        pos = positions[i-1]
        signal = cdata.x[pos[1].type(torch.LongTensor)]
        signal = torch.reshape(signal,( len(signal),1))
        data.x = signal # + (0.01**0.5)*torch.randn(len(signal),1) #random noise

        nodeErrors = output[pos[1].type(torch.LongTensor)] - model.forward(data)
        L2Error = torch.sqrt(1/len(nodeErrors)*torch.sum(torch.pow(nodeErrors,2)))
        L2Errors.append(L2Error)
        end = time.time()
        print(f"{i}: Took {(end-start)* 1000.0:.3f} ms")

    err = [x.detach().numpy() for x in L2Errors]

    return err

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

def fct_eval(fct, pos): 
    """
    For evaluating the signal we got above in a vector in [0,1]^2. The idea is that the IFFT gives us a 256x256
    grid. For every 2D vector, we check by the above calculations in which grid-patch it lies and then
    evealuate the signal we receive by IFFT in it.
    """
    
    i = pos[0][:,0]*257-1
    i = i.type(torch.LongTensor) 
    j = pos[0][:,1]*257-1
    j = j.type(torch.LongTensor)
    return fct[i,j]

ifft2_real = create_randn_bandlimited()
signal = fct_eval(ifft2_real, positions[-1]).reshape(2**N,1) #evaluation in all vectors we sampled
errs = [ ]
for i in [1, 5, 9]:
    errs.append(error_fct(i, signal))
    
xAxis = [2**n for n in range(1,14)]
fig = plt.figure()
plt.xlabel('Nodes')
plt.ylabel('l2error')
plt.yscale('log')
#txt="radius: " + str((radius)/10)
#plt.figtext(0.5, 1, txt, wrap=True, horizontalalignment='center', fontsize=15)
plt.plot(xAxis,errs[0],label='0.1')
plt.plot(xAxis,errs[1], label='0.5')
plt.plot(xAxis,errs[2],label='0.9')
plt.legend()
fig.savefig('../output/BLRandnSignalGraphSage2MLPl2Error' + str(2**N) + 'Nodes.png', dpi=fig.dpi)



slope0, intercept0 = np.polyfit(np.log(xAxis), np.log(errs[0]), 1)
slope1, intercept1 = np.polyfit(np.log(xAxis), np.log(errs[1]), 1)
slope2, intercept2 = np.polyfit(np.log(xAxis), np.log(errs[2]), 1)

xAxis = [2**n for n in range(1,14)]
fig = plt.figure()
plt.ylabel('loglog')
#txt="radius: " + str((radius)/10) + "  |  slope: " + str(slope)
#plt.figtext(0.5, 1, txt, wrap=True, horizontalalignment='center', fontsize=15)
plt.loglog(xAxis[:],errs[0], '--', label = '0.1 | slope: ' + str(slope0))
plt.loglog(xAxis[:],errs[1], '--', label = '0.5 | slope: ' + str(slope1))           
plt.loglog(xAxis[:],errs[2], '--', label = '0.9 | slope: ' + str(slope2))           
plt.legend()

#fig.savefig('../output/Logl2Error' + str(1002) + 'Nodes.png', dpi=fig.dpi)
fig.savefig('../output/BLRandnSignalGraphSage2MLPl2Error' + str(2**N) + 'Nodes.png', dpi=fig.dpi)


    
