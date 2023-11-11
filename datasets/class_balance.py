import pickle
import numpy as np 
import torch 

with open('haldane_bundle.pkl', 'rb') as f:
    data = pickle.load(f)
    
    
chern = np.array([x['chern'] for x in data])

# data is already randomized so just take first 2.5k of each
# note that some might have less than 2.5k examples
chern0 = [i for i in range(len(data)) if chern[i] == 0][0:2500]
chern1 = [i for i in range(len(data)) if chern[i] == 1][0:2500]
chern_1 = [i for i in range(len(data)) if chern[i] == -1][0:2500]
chern2 = [i for i in range(len(data)) if chern[i] == 2][0:2500]
chern_2 = [i for i in range(len(data)) if chern[i] == -2][0:2500]


idxs = chern0 + chern1 + chern_1 + chern2 + chern_2 
class_balanced = [data[i] for i in idxs]
pickle.dump(class_balanced, open("class-balanced.pkl", "wb"))

