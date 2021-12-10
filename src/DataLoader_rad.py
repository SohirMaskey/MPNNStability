import numpy as np
import torch
from torch_cluster import radius_graph #to build random geometric graphs, given the positions in 2D and the radius
from torch_geometric.data import Data #build Data objects in pytorch
import time


import random

import os.path as osp #to open data 

from torch_geometric.data import Dataset, download_url

import os #to save data

class RGGDataset_rad(Dataset):
    
    def __init__(self, root, roottransform=None, pre_transform=None, size=1000, skip=10,  radius=0.5):
        """
        root = where the dataset should be stored. 
        This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data)
        
        size = The size of the largest graph which is later used as the limit object
        
        skip = We build every skip'th graph,
        i.e., if size=1000, skip=10, we build graphs with node size 10,20,30,..,990,1000.

        """
        #super(RGGDataset, self).__init__(root, transform, pre_transform)
        self.size = size 
        self.skip = skip
        self.root = root
        self.radius = radius
        self.number = int(self.size/self.skip)
    
    
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered."""
        
        return [f'graph_r{int(radius*10)}_{int(i*self.skip)}.pt' for i in range(1,self.number)]
    
    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        
        return [f'graph_r{int(self.radius*10)}_{int(i*self.skip)}.pt' for i in range(1, self.number)]
    
    def download(self):
        """
        1. build a tensor of size (self.size,2) random vectors in [0,1]^2
        2. sample randomly vector of size self.skip, 2*self.skip, 3*self.skip, ..., self.size-self.skip
        3. append everything to a list and save it in /self.root/raw
        """
        
        if osp.isfile(self.root + '/raw/positions_{self.size}_{self.skip}.pt'):
            print('already downloaded')
            pass
        #we dont really download anything graph_signal_10001.pickle
        pos = torch.rand(self.size,2)
        positions = [ ]
        for i in range(self.skip,self.size,self.skip):
            p = torch.ones(len(pos))*1/len(pos)
            idx = p.multinomial(num_samples=i, replacement=False)
            b = pos[idx]
            positions.append([b, idx])
        positions.append([pos, torch.range(0,self.size-1)])
        torch.save(positions, 
            os.path.join(self.raw_dir, 
                f'positions_{self.size}_{self.skip}.pt'))
        #print('triggered')
        return [f'positions_{self.size}_{self.skip}.pt']
    
    """
    def download(self):
        pass
    """    

    def process(self): 
        """
            Use the positions given by download() and create a sequence of graphs by the Random Geometric 
            Graph Alg with radius.size
        """
        
        positions = torch.load(osp.join(self.raw_dir, f'positions_{self.size}_{self.skip}.pt'))
        
        for i in range(1,self.number+1):
            print(i)
            print(len(positions[i-1][0]))
            batch = torch.zeros(int(i*self.skip)).type(torch.LongTensor)
            edge_index = radius_graph(positions[i-1][0], r=self.radius, batch=batch, loop=False)
            graph  = Data(edge_index = edge_index)
            #graphs.append(graph)
            torch.save(graph, 
                os.path.join(self.processed_dir, 
                    f'graph_r{int(self.radius*10)}_{int(i*self.skip)}.pt'))
            #nx.draw(to_networkx(graph))
            
    def len(self):
        return len(self.size)

    def get(self, radius, graphsize):
        """
            Given the radius and the graphsize, we readout the graphs in "self.root/processed"
        """
        
        data = torch.load(osp.join(self.processed_dir, 'graph_r{radius}_{graphsize}.pt'.format(radius=radius,
                                                                                               graphsize=graphsize)))
        return data
    
    
    
if __name__=='__main__':
    DL = RGGDataset(root = '//home/groups/ai/maskey/input_rad')
    #DL.download()
    DL.process()