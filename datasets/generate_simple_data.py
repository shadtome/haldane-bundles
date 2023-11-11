import sys
import os
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0,parent)
from linebundles import device 
import haldane_bun as hb
import torch
import numpy as np
import pickle as pkl
import progressbar



"""
Fix functions:

$H_x = -3 \cos( 2\pi (-2x + 2y)) -3 \cos( 2\pi (-1x - 2y)) -4\cos( 2\pi (-y))  $ 

$H_y = -3 \sin( 2\pi (-2x + 2y)) -3 \sin( 2\pi (-1x - 2y)) -4\sin( 2\pi (-y))  $ 

$H_z = t_1\sin(2\pi (-2x-y)) + t_2\sin(2\pi(x-y)) + t_3\sin(2\pi(x)) $

And vary $t_1, t_2, t_3$ over $[-3,3]^3$ to generate the dataset.

These functions were chosen randomly.

"""

#progress bar
amount = len(np.arange(-3,3,6/22))
widgets = [ progressbar.Bar('*'), '(', progressbar.ETA(), ')']
total_steps =np.power(amount,3)+1
bar = progressbar.ProgressBar(total_steps,widgets=widgets).start()

data = []

min_deg = -4
max_deg = 4
d = max_deg - min_deg + 1

n = 32 # resolution to sample vector bundle at 

coef_c = torch.zeros((d,d))
coef_c[2,6]= -3.
coef_c[3,2]= -3.
coef_c[4,3]= -4.
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, 6/22)

for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            coef_r = torch.zeros((2,d,d))
            coef_r[1,2,3] = t1
            coef_r[1,5,3] = t2
            coef_r[1,5,4] = t3
            
            f_r = [ {'coef': coef_r, 'min_d':-min_deg, 'center' : 0}]
            
            example = hb.haldane_bun(f_r, f_c)
    
            # check for interal consistency before adding to the dataset 
            c1  = example.Chern_number(1000,1)
            c2  = example.Chern_number(2000,1)
            if c1 - c2 == 0:
                c3 = example.Chern_number(3000,1)
                if c3 - c2 == 0:

                    chern_number = int(c3)
                    sampled_bundle = example.sample(n)

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number }]
            bar.update(bar.currval+1)

 
# save the data
pkl.dump(data, open("haldane_simple.pkl", "wb"))

