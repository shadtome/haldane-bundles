*This is code supporting the paper 'Haldane Bundles: A Dataset for Learning to Predict the Chern Number of Line Bundles on the Torus'*

# Generate Haldane Bundles

The purpose of this code base is to generate random complex line bundles over the 2-torus and to calculate 
the Chern number for each of these efficiently. We then generated a large number of these line bundles to 
create a dataset appropriate for training and evaluating machine learning models. 
 
To generate each line bundle we define a Haldane pair (*G*,*F*) of Fourier polynomials (*G* is a real-valued 
Fourier polynomial and *F* is a complex-valued Fourier polynomial). *F* and *G* parametrize a line bundle on the torus.  
We are able to easily calculate the Chern number of the resulting complex line bundles using the coefficients that define *F* and *G*.

The code base below generates random Haldane pairs to give a random line bundle, takes tensor products of 
the resulting line bundles, outputs images representing the line bundle, and approximates the Chern number.  
Efficient calculation of Chern numbers is enabled through the use of PyTorch and GPUs. For a Haldane pair (*G*,*F*), 
we need to make sure that *G* and *F* have no common zeros on the torus, otherwise 
the corresponding line bundle may not be defined over the whole torus.  We do not have a way to detect this, except 
by computing the Chern number for different ways of partitioning the torus and seeing if it converges.  

## Background
Knowledge of smooth manifolds, differential forms, connections, Chern-Weil theory, vector bundles, 
and Fourier polynomials is helpful in understanding the mathematics underlying the calculations and concepts. Knowing something 
about topological insulators can provide motivation for the (non-trivial) mathematics.

## Features

- Generate Haldane bundles by either:
   - randomly generating the coefficients for *G* and *F*,
   - specifying your own coefficients for *G* and *F*.
- Calculate the Chern number using PyTorch.
- Generate tensor products of Haldane bundles.
- Output an image of the coordinates of the vectors representing a line bundle.

### To Add

- Include a method to determine if a pair (*G*,*F*) is a Haldane pair

## Dataset Creation

Code to generate the Haldane Bundle dataset in our paper is `datasets/generate_data.py`. For each example in this dataset, 
a new Haldane pair (*G*,*F*) is generated and we calcualte the Chern number and sample the corresponding line bundle. 
The final dataset, a downsampled version of the original data which balances classes is constructed in `datasets/class_balance.py`.

We also include code in `datasets/generate_intermediate_data.py` which generates a simpler version of the dataset. 
In particular, we randomly selected and fixed 10 forms of Haldane pairs (*G*,*F*) and for each pair, vary the coefficients of *G* to generate the dataset. 

Finally, we include code in `datasets/generate_simple_data.py` for a dataset constructed from a single form of Haldane pair (*G*,*F*),
varying the coefficients on *G* alone. This dataset is a straightforward machine learning task and 100 percent test accuracy can be achieved with a range of
off-the-shelf architectures.


Dependencies
---------
- [PyTorch](https://pytorch.org/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [numPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [progressbar](https://pypi.org/project/progressbar2/)



## Usage for linebundles

```python
from haldanebun.linebundles import FourierBun as fb

# Example:
# Generates a single random Haldane pair (G,F)
# with the following parameters
num_pairs = 1
r_neg = 3
r_pos = 3
c_neg = 3
c_pos = 3
center_r =0
center_c =0
bun = fb.Fourier_Bundle.random_bundle(num_pairs,r_pos,r_neg, c_pos, 
                                         c_neg,center_r,center_c)

# outputs an image of one of the components of the
# line bundle. gamma and lamb are the parameters (0 or 1)
# for the various components for the vectors of this
# line bundle.
part_size = 100
gamma = 0
lamb = 1
bun.output_image(gamma,lamb,part_size)

# Calculates the Chern number with respect to the 
# partition size of the torus and batch size.
part_size = 3000
batch_size = 1
print(f'Chern number = {bun.Chern_number(part_size,batch_size)}')

# Print the Haldane pairs (G,F)
index=0
bun.output_Fourier_poly(index)


 
```
## Authors

- [Cody Tipton](https://github.com/shadtome)
- [Elizabeth Coda]()
- [Davis Brown]
- [Alyson Bittner]
- [Henry Kvinge]

Notice
------
The research described in this work is part of the Mathematics of Artificial Reasoning in Science (MARS) Initiative at Pacific Northwest National Laboratory (PNNL).  It was conducted under the Laboratory Directed Research and Development Program at PNNL, a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy.

Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

```
         PACIFIC NORTHWEST NATIONAL LABORATORY
                  operated by
                   BATTELLE
                   for the
           UNITED STATES DEPARTMENT OF ENERGY
            under Contract DE-AC05-76RL01830
```

License
-------

Released under the 3-Clause BSD license (see License.rst)
# haldane-bundles
