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



min_deg = -4
max_deg = 4
d = max_deg - min_deg + 1

n = 32 # resolution to sample vector bundle at 

# amount of data
amount = 50000

#set up progress bar
widgets = [ progressbar.Bar('*'), '(', progressbar.ETA(), ')']
total_steps = len(amount)+1
bar = progressbar.ProgressBar(total_steps,widgets=widgets).start()

data = []
for i in range(amount):
    coef_c = torch.zeros((d,d))
    coef_r = torch.zeros((2,d,d))
    
    # randomly select sparse, integer coefficients for F 
    N = np.random.choice(range(2,7))
    for i in range(N):
        j,m = np.random.choice(range(d),2)
        coef_c[j,m] = np.random.choice(np.arange(-8,9))
    
    # randomly select sparse, integer coefficients for G 
    M = np.random.choice(range(1,7))
    for i in range(M):
        j,m = np.random.choice(range(d),2)
        p = np.random.choice([0,1])
        coef_r[p,j,m] = np.random.choice(np.arange(-8,9))
        
    f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]
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
pkl.dump(data, open("haldane_bundle.pkl", "wb"))
