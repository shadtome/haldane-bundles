import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0,parent)
from linebundles import FourierBun as fb
import torch
import numpy as np


class haldane_bun(fb.Fourier_Bundle):
    """Child class of Fourier_Bundle specifically designed for 
    the dataset generated for the paper.  It only generates one
    Haldane pair (G,F), unlike the parent class which can take 
    a list a generate a tensor product of them.  It also has its 
    own sample function, which outputs the represenative of the
    information needed for the generated data set in the 
    paper."""
    def __init__(self,r_fourier : map, c_fourier : map):
        super().__init__(r_fourier,c_fourier)

    @classmethod
    def random_bundle(cls, r_pos: list, r_neg: list,
                       c_pos: list, c_neg: list, center_r=0, center_c=0):
        return super().random_bundle(1, r_pos, r_neg, c_pos, c_neg, center_r, center_c)

    def sample(self,part_size : int):
        """
        returns \psi = [R,F] sampled over the torus, at a resolution (part_size, part_size)

        """
        index = 0
        F=self.F[index]
        G=self.G[index]
        Max_1_F = F['coef'].size(0)
        Max_2_F = F['coef'].size(1)

        Max_1_G = G['coef'].size(1)
        Max_2_G = G['coef'].size(2)


        # sample F and G over the torus
        sampled_F = torch.zeros((part_size,part_size)) + 1j*torch.zeros((part_size,part_size))
        sampled_G = torch.zeros((part_size,part_size))


        px = np.arange(0, 1, 1/part_size)
        py = np.flip(px)
        torus = np.zeros((part_size,part_size,2))
        for i, px_ in enumerate(px):
            for j, py_ in enumerate(py):
                torus[j,i] = (px_,py_) # [j,i] ordering is for plt.imshow purposes
        torus = torch.tensor(torus)


        for k_1 in range(Max_1_F):
            for k_2 in range(Max_2_F):
                if F['coef'][k_1][k_2]!=0:
                    sampled_F += 1/(2*np.pi)*F['coef'][k_1][k_2]*torch.exp(2*np.pi*1j*((k_1-F['min_d']+F['center'])*torus[:,:,0] + (k_2-F['min_d']+F['center'])*torus[:,:,1])) 


        Max_1_G = G['coef'].size(1)
        Max_2_G = G['coef'].size(2)
        for k_1 in range(Max_1_G):
            for k_2 in range(Max_2_G):
                if G['coef'][0][k_1][k_2]!=0:
                    sampled_G +=  1/(2*np.pi)*G['coef'][0][k_1][k_2]*torch.cos(2*np.pi*((k_1-G['min_d']+G['center'])*torus[:,:,0] + (k_2-G['min_d']+G['center'])*torus[:,:,1]))
                if G['coef'][1][k_1][k_2]!=0:
                    sampled_G += 1/(2*np.pi)*G['coef'][1][k_1][k_2]*torch.sin(2*np.pi*((k_1-G['min_d']+G['center'])*torus[:,:,0] + (k_2-G['min_d']+G['center'])*torus[:,:,1]))



        sampled_R = sampled_G + torch.sqrt((sampled_G**2 + sampled_F*torch.conj_physical(sampled_F)))

        return torch.stack((sampled_R, sampled_F))