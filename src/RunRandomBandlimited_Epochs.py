"""
We created a dataset with graphs of size 2^1,2^2,..,2^15,2^16. This data set is saved in //home/groups/ai/maskey/input_rad/processed. A graph of size n and constructed with radius r, can be read out by "DL.get(10*r,n)" after constructing the Data loader object, e.g. "DL = RGGDataset_grid(root = '../../input_rad')".

Then, we use the finest graph(size 2^16) as the continuous limit object and calculate graph wise l^2 errors with some graph signal. 

"""

import numpy as np
import torch
from torch_geometric.data import Data #for constructing data/graph objects from the sample points
import time

import random

import matplotlib.pyplot as plt

from torch_geometric.nn import SAGEConv #used for the model construction
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

import sys
sys.path.insert(1,'../src')
from DataLoader_rad_grid import RGGDataset_grid #This is the class to load data
from TwoLayerGraphSage import GCN #This is our MPNN

import os.path as osp

from torch_geometric.data import Dataset, download_url

import os

import pickle #for saving

#DL = RGGDataset_grid(root = '../../input_rad')

#dataset = DL.get(1,2**5)

#load the positions, list of positions for all graphs
positions = torch.load('../../input_rad/raw/grid_positions_128.pt')

model = GCN()
model.load_state_dict(torch.load( '../models/GCNTwoLayersGraphSage'))

N = 14


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

def error_fct(radius, signal):
    """
    Calculates for a given radius and signal the error between the coarser graphs and finest graph
    
    """
    DL = RGGDataset_grid(root = '../../input_rad', radius= radius/10,size = 7)
    DL.process_grid()
    
    L2Errors = []

    cdata = DL.get(radius,2**N)
    cpos = positions[-1]

    cdata.x = signal 

    output = model.forward(cdata)

        
    for i in range(1, N):
        data = DL.get(radius, 2**i) 
        pos = positions[i-1]
        signal = cdata.x[pos[1].type(torch.LongTensor)]
        signal = torch.reshape(signal,( len(signal),1))
        data.x = signal # + (0.01**0.5)*torch.randn(len(signal),1) #random noise

        nodeErrors = output[pos[1].type(torch.LongTensor)] - model.forward(data)
        L2Error = torch.sqrt(1/len(nodeErrors)*torch.sum(torch.pow(nodeErrors,2)))
        L2Errors.append(L2Error)

    err = [x.detach().numpy() for x in L2Errors]

    return err

errs = [ [0]*2**N, [0]*2**N, [0]*2**N]

#low_pass = lambda x:  (1+(torch.tensor(x[:,0]**2 + x[:,1]**2))).pow_(-1)

low_pass = lambda x:  x[:,0]*x[:,1] #the signal

#y = torch.randn(2**N,1)
#cdata = DL.get(1,2**N)
cpos = positions[-1]
signal = low_pass(cpos[0])
signal = torch.reshape(signal,( len(signal),1))
#cdata.x = y 
#signal = y

epochs = 10 #It is a random experiment, How often do we repeat it?

ifft2_real = create_randn_bandlimited()
signal = fct_eval(ifft2_real, positions[-1]).reshape(2**N,1) #evaluation in all vectors we sampled

for j in range(epochs):
    start = time.time()
    ifft2_real = create_randn_bandlimited()
    signal = fct_eval(ifft2_real, positions[-1]).reshape(2**N,1) 
    #error = [ , , , ]
    for i, value in enumerate([1,5,9]):    
        errs[i] = [sum(x) for x in zip(errs[i], error_fct(value, signal))]
        #error = error + error_fct(i, signal) 
        errs[i] = [x/epochs for x in errs[i]]
        #errs.append([x/epochs for x in error_fct(i, signal)])
    #errs = [sum(x) for x in zip(errs, errs)]
    end = time.time()
    print(str(j) + ": " +  str((end-start)*1000))
    
with open('../output/BLRandnSignalGraphSage2MLP' + str(2**N) + 'Nodes' + '.pickle', 'wb') as output:
    pickle.dump(errs, output)
    
    
    
xAxis = [2**n for n in range(1,N)]
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
fig.savefig('../output/BLRandnSignalGraphSage2MLPl2Error' + str(2**N) + 'Nodes.png', dpi=600)
plt.show()

slope0, intercept0 = np.polyfit(np.log(xAxis), np.log(errs[0]), 1)
slope1, intercept1 = np.polyfit(np.log(xAxis), np.log(errs[1]), 1)
slope2, intercept2 = np.polyfit(np.log(xAxis), np.log(errs[2]), 1)



xAxis = [2**n for n in range(1,N)]
fig = plt.figure()
#plt.xlabel('Nodes')
plt.ylabel('loglog')
#txt="radius: " + str((radius)/10)
#plt.figtext(0.5, 1, txt, wrap=True, horizontalalignment='center', fontsize=15)
#txt="radius: " + str((radius)/10) + "  |  slope: " + str(slope)
#plt.figtext(0.5, 1, txt, wrap=True, horizontalalignment='center', fontsize=15)
plt.loglog(xAxis[:],errs[0], '--', label = '0.1 | slope: ' + str(slope0))
plt.loglog(xAxis[:],errs[1], '--', label = '0.5 | slope: ' + str(slope1))           
plt.loglog(xAxis[:],errs[2], '--', label = '0.9 | slope: ' + str(slope2))           
plt.legend()

fig.savefig('../output/BLRandnSignalGraphSage2MLPLogl2Error' + str(2**N) + 'Nodes.png', dpi=600)
plt.show()
