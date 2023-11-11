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


#set up progress bar
type_1 = len(np.arange(-3,3,6/10))
type_2 = len(np.arange(-3,3,6/32))
widgets = [ progressbar.Bar('*'), '(', progressbar.ETA(), ')']
total_steps =9*np.power(type_1,3)+np.power(type_2,2)+1
bar = progressbar.ProgressBar(total_steps,widgets=widgets).start()


min_deg = -4
max_deg = 4
d = max_deg - min_deg + 1

n = 32 # resolution to sample vector bundle at 

# This changes 9 out of the 10 example generators below.  This 
# determines the step in "np.arange(-3,3,step_1)"
step_1 = 6/10

# This changes 1 of the example generators below.  This
# determines the step in "np.arange(-3,3,step_2)"
step_2 = 6/32

data = []

###### (F,G) pair 0 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0., -4.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  8.,  4.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]



t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[1,6,2] = t1
            coef_r[1,6,6] = t2
            coef_r[1,7,4] = t3
            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':0 }]
            bar.update(bar.currval+1)
                   
                    
###### (F,G) pair 1 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  5.,  0.,  0., -5., -2.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[1,2,5] = 8
            coef_r[1,3,8] = -8
            coef_r[0,4,6] = t1
            coef_r[1,7,2] = t2
            coef_r[0,8,7] = t3
            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':1 }]   
            bar.update(bar.currval+1)     
                    
                   
                    
###### (F,G) pair 2 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 5.,  0.,  0.,  8.,  0.,  8.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -6.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -8., -6.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[1,1,3] = t1
            coef_r[0,4,2] = t2
            coef_r[1,5,5] = 5
            coef_r[0,5,7] = t3
            coef_r[1,5,8] = 6
            coef_r[1,6,7] = -6
            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':2 }]
            bar.update(bar.currval+1)

###### (F,G) pair 3 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  6.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -4.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[0,0,5] = t1
            coef_r[0,5,1] = t2
            coef_r[1,0,8] = t3
            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':3 }]                    
            bar.update(bar.currval+1)                
                   
###### (F,G) pair 4 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  4.,  0.,  0.,  0., -6.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-6.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[1,1,3] = t1
            coef_r[1,2,5] = t2
            coef_r[0,3,5] = t3
            coef_r[1,6,5] = -5
            coef_r[1,6,7] = 8
            coef_r[0,7,8] = 6

            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':4 }]                    
            bar.update(bar.currval+1)         

###### (F,G) pair 5 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0., -4.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  8.,  4.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[1,6,2] = t1
            coef_r[1,6,6] = t2
            coef_r[1,7,4] = t3

            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':5 }]                    
            bar.update(bar.currval+1)

###### (F,G) pair 6 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -6.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0., -6., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_2)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[0,7,4] = t1
            coef_r[1,8,6] = t2
  
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':6 }]                    
            bar.update(bar.currval+1)  

###### (F,G) pair 7 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -4.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  4.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -7.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[1,1,7] = t1
            coef_r[0,4,1] = t2
            coef_r[1,4,8] = 7
            coef_r[0,5,8] = -7
            coef_r[1,8,0] = t3

            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':7 }]                    
            bar.update(bar.currval+1)                            

###### (F,G) pair 8 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -4.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  7.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  7.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[0,3,3] = -4
            coef_r[1,4,0] = -4
            coef_r[0,5,3] = t1
            coef_r[0,7,2] = t2
            coef_r[1,8,1] = t3

            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':8 }]                    
            bar.update(bar.currval+1)        

###### (F,G) pair 9 ###### 
coef_c = torch.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0., -6.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0., -4.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -3.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
f_c = [ {'coef': coef_c, 'min_d':-min_deg, 'center' : 0}]

t_range = np.arange(-3,3, step_1)
for i, t1 in enumerate(t_range):
    for j, t2 in enumerate(t_range):
        for m, t3 in enumerate(t_range):
            
            coef_r = torch.zeros((2,d,d))
            coef_r[1,0,0] = t1
            coef_r[1,1,1] = t2
            coef_r[1,6,4] = t3
            coef_r[0,7,5] = -7

            
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

                    data += [{'coef_c': coef_c, 'coef_r': coef_r, 'line_bundle': sampled_bundle, 'chern':chern_number, 'haldane_pair':9 }]                    
            bar.update(bar.currval+1)                               
pkl.dump(data, open("haldane_intermediate.pkl", "wb"))                      