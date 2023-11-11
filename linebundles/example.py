import FourierBun as fb
import torch
import numpy as np



# Example 1:
# Generates a single random Haldane pair (G,F)
# With complex Fourier polynomial F with coefficients 
# in degrees (-c_neg+center_c,-c_neg+center_c) 
# to (c_pos+center_c,c_pos+center_c)
# with 'center' around the index (center_c, center_c)
# and a real polynomial G with coefficients 
# in degrees (-r_neg + center_r,-r_neg + center_r) 
# to (r_pos + center_r,r_pos + center_r)
# with "center" around the index (center_r,center_r)
# Then it constructs eigenvectors 
#
#       Psi = [G + sqrt(G^2+ FF*) , F]    and    Psi^{daggger} = [-F*, G - sqrt(G^2 + FF*)]
# where * is conjugate.
# with the following parameters
print('Example 1')
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
bun.output_image(gamma,lamb,part_size,normalize=True)
print(bun.output_rep(32,False))


# Calculates the Chern number with respect to the 
# partition size of the torus and batch size.
part_size = 3000
batch_size = 1
print(f'Chern number = {bun.Chern_number(part_size,batch_size)}')

# Print the Haldane pairs (G,F)
index=0
bun.output_Fourier_poly(index)

#Example 2
# Take the line bundle generated above and take 
# a finite number of tensor products of it.
print('Example 2')
num_pairs = 5
r_fourier = []
c_fourier = []
for i in range(num_pairs):
    r_fourier.append(bun.G[0])
    c_fourier.append(bun.F[0])
bun_tensor_n = fb.Fourier_Bundle(r_fourier,c_fourier)

part_size = 100
gamma = []
lamb = []
for i in range(num_pairs):
    gamma.append(np.random.randint(2))
    lamb.append(np.random.randint(2))
bun_tensor_n.output_image(gamma,lamb,part_size)

part_size = 3000
batch_size = 1
print(f'Chern number = {bun_tensor_n.Chern_number(part_size,batch_size)}')

# Example 3
# Generate a tensor product of different random 
# Haldane bundles.
# The parameters are lists corresponding to each pair
print('Example 3')
num_pairs = 5
r_neg = []
r_pos = []
c_neg = []
c_pos = []
center_r = []
center_c = []

for i in range(num_pairs):
    r_pos.append(np.random.randint(0,10))
    r_neg.append(np.random.randint(0,10))
    c_pos.append(np.random.randint(0,10))
    c_neg.append(np.random.randint(0,10))
    center_r.append(np.random.randint(-10,10))
    center_c.append(np.random.randint(-10,10))

tensor_rand_bun = fb.Fourier_Bundle.random_bundle(num_pairs,r_pos,r_neg,
                                             c_pos,c_neg,center_r,center_c)

# The gamma and lamb are lists of 0's or 1's for tensor product
# of Haldane bundles.
gamma = [0,0,1,0,1]
lamb = [1,0,0,1,0]
part_size = 100
tensor_rand_bun.output_image(gamma,lamb,part_size)

part_size = 5000
batch_size = 1
print(f'Chern number = {tensor_rand_bun.Chern_number(part_size,batch_size)}')


# Example 4
# Construct a specific line bundle by 
# explicitly giving the coefficients
print('Example 4')
def ex_gen(M,t_1,t_2):
    c_coef = torch.zeros(size=[3,3])
    c_coef[2][2]=t_1
    c_coef[1][0]=t_1
    c_coef[0][1]=t_1
    r_coef = torch.zeros(size=[2,5,5])
    r_coef[0][2][2]=M
    r_coef[1][4][3]=2*t_2
    r_coef[1][1][0]=2*t_2
    r_coef[1][1][3]=2*t_2
    comp_map = {'coef':c_coef, 'min_d': 1, 'center' : 0}
    real_map = {'coef':r_coef, 'min_d': 2, 'center' : 0}

    return fb.Fourier_Bundle(r_fourier=real_map,c_fourier=comp_map)

# Can change the line bundle with tese parameters
t_2=3
M=10
ex=ex_gen(M=M,t_1=1,t_2=t_2)
part_size=100

# The pair of Fourier polynomials (G,F)
ex.output_Fourier_poly(0)

#Images of the components
ex.output_image(gamma=0,lamb=0,part_size=part_size,normalize=True)
ex.output_image(gamma=0,lamb=1,part_size=part_size,normalize=True)

# Chern number changes depending on the parameters.
print(f'The Chern number for when {t_2/M} relative to 1/3sqrt(3)={1/(3*np.sqrt(3))}')
print(f' is {ex.Chern_number(1000,1)}')





    



