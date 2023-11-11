import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0,parent)
from linebundles import device
   



class Fourier_Bundle:
    """Constructs Line bundle corresponding to a 
    Haldane sequences ((G_1,F_1),....,(G_n,F_n)) of 
    Fourier polynomials over the torus, with methods to compute the 
    Chern number, output the Fourier polynomials, and output components 
    of the vectors associated with it.
    
    Can directly input the list of real Fourier polynomials and complex
    Fourier polynomials to construct the Haldane pairs, or have it 
    randomly construct a Haldane sequence."""
    def __init__(self,r_fourier : list , c_fourier : list):
        """
        r_fourier: list of dictionaries with keys 'coef', 'min_d', and 'center'.
         The key 'coef' has value of a tensor of shape [2,N,N] of 
         coefficients for the real Fourier polynomial (coefficients for 
         cos(x) and sin(x)), where N is the minimal positive integer so 
         that this tensor contains all the coefficients. Let min_d, 
         max_d be positive integers and center_r be any integer.
         The tensor corresponding to the coefficients is a block matrix
         with the upper left entry correspond to the coefficient with 
         index (-min_d+center_r, -min_d+center_r) and the bottom right
         corresponds to the coefficient with index 
         (max_d+center_r,max_d+center_r).  Hence, the values for
         'min_d' is min_d and the value for 'center' is center_r

        c_fourier: list of dictionary with keys 'coef', 'min_d', and 'center'.
         The key 'coef' has value that is a tensor of shape [N,N] of
         coefficients for the complex Fourier polynomial, where
         N is the minimal positive integer so that this tensor
         contains all the coefficients. 
         Let min_d, max_d be positive integers and center_c be any 
         integer. The tensor corresponding to the coefficients is a 
         block matrix with the upper left entry correspond to the 
         coefficient with index (-min_d+center_c, -min_d+center_c) 
         and the bottom right corresponds to the coefficient with index 
         (max_d+center_c,max_d+center_c).  Hence, the values for
         'min_d' is min_d and the value for 'center' is center_c
         """
        
        # We need to handle if the input is a list
        # or just maps. 
        c_fourier_list = []
        r_fourier_list = [] 
        if isinstance(c_fourier, dict):
            c_fourier_list.append(c_fourier) 
        else:
            c_fourier_list = c_fourier
            
        if  isinstance(r_fourier,dict):
            
            r_fourier_list.append(r_fourier)
        else:
            
            r_fourier_list = r_fourier
        if len(c_fourier)!=len(r_fourier):
            ex = f'c_fourier and r_fourier must be both of the same length,'
            ex += f' instead we got lengths {len(c_fourier)} and {len(r_fourier)}.'
            raise Exception(ex)

        self.F=c_fourier_list
        """List of complex valued Fourier polynomials"""

        self.G=r_fourier_list
        """List of real-valued Fourier polynomials"""

        self.n = len(self.F)
        """ This is the number of tensor products"""

        self.k_G, self.k_F = self._k_tensor_box()
        """Tensor of shape [N,N,2] of indices (k_1,k_2)"""

        self.k_G_line, self.k_F_line = self._k_tensor_line()
        """tensors of shape [N] with values (-min_d+center,..., 
        max_d+center)"""

    @classmethod
    def random_bundle(cls,num_pairs : int,r_pos : list,r_neg : list,
                      c_pos : list,c_neg : list, center_r = 0, center_c = 0):
        """ Generates random Haldane pairs of Fourier polynomials.
        
        num_pairs: the number of Haldane pairs
        
        r_pos: list of positive integers for real-valued Fourier 
        polynomials representing max positive degree from (0,0).
        r_neg: list of positive integers for the real-valued Fourier
            polynomials representing max negative degree from (0,0).
        c_pos: list of positive integers for complex-valued Fourier
            polynomials representing max positive degree from (0,0).
        c_neg: list of positive integers for complex-valued Fourier
            polynmials representing max negative degree from (0,0).
        center_r: (default = 0) list of integers for real-valued Fourier 
            polynomials representing the shift from the index (0,0).
        center_c: (default = 0) list of integers for complex-valued 
            Fourier polynomials representing the shift from the 
            index (0,0)."""
        

        # make sure the inputs are either lists of the correct size 
        # and of the correct ranges of ints
        deg = {'r_neg' : r_neg, 'r_pos' : r_pos, 'c_neg': c_neg, 'c_pos' : c_pos}
        for key in deg:
            value = deg[key]
            if isinstance(value,int):
                if 0<=value:
                    value_list = []
                    for i in range(num_pairs):
                        value_list.append(value)
                    deg[key]= value_list
                else: 
                    raise Exception(f'{key} must be a int greater or equal to 0. Instead we got {value}')
            elif len(value) != num_pairs:
                raise Exception(f'{key} must be a list of length {num_pairs}, got {value} of length {len(value)}')      

        centers = {'center_r': center_r, 'center_c' : center_c}
        for key in centers:
            value = centers[key]
            if isinstance(value,int):
                value_list = []
                for i in range(num_pairs):
                    value_list.append(value)
                centers[key]=value_list
            elif len(value) != num_pairs:
                raise Exception(f'{key} must be a list of length {num_pairs}, got {value} of length {len(value)}')      
        
        f_c = []
        f_r = []

        for i in range(num_pairs):
            # construct the random Haldane pair for each i.
            fourier_r,fourier_c=gen_fourier(deg['r_pos'][i],deg['r_neg'][i],
                                            deg['c_pos'][i],deg['c_neg'][i],
                                            centers['center_r'][i],centers['center_c'][i])
            f_c.append(fourier_c)
            f_r.append(fourier_r)
        
        return cls(f_r,f_c)
   
    
    @classmethod
    def random_sparse_bundle(cls,num_pairs : int,r_pos : list,r_neg : list,
                             c_pos : list,c_neg : list,num_coef_r : list,
                             num_coef_c : list, center_r = 0, center_c = 0):
        """ Generates random Haldane pairs of sparse Fourier 
        polynomials.
        
        num_pairs: the number of Haldane pairs
        
        r_pos: list of positive integers for real-valued Fourier 
            polynomials representing max positive degree from (0,0).
        r_neg: list of positive integers for the real-valued Fourier
            polynomials representing max negative degree from (0,0).
        c_pos: list of positive integers for complex-valued Fourier
            polynomials representing max positive degree from (0,0).
        c_neg: list of positive integers for complex-valued Fourier
            polynmials representing max negative degree from (0,0).
        num_coef_c: list of integers corresponding to max number 
            of non-zero coefficients for F
        num_coef_r: list of integers corresponding to max number 
            of non-zero coefficients for G
        center_r: (default = 0) list of integers for real-valued Fourier 
            polynomials representing the shift from the index (0,0).
        center_c: (default = 0) list of integers for complex-valued 
            Fourier polynomials representing the shift from the 
            index (0,0)."""



        # make sure the inputs are either lists of the correct size 
        # and of the correct ranges of ints
        deg = {'r_neg' : r_neg, 'r_pos' : r_pos, 'c_neg': c_neg, 'c_pos' : c_pos, 
               'num_coef_r' : num_coef_r, 'num_coef_c' : num_coef_c}
        for key in deg:
            value = deg[key]
            if isinstance(value,int):
                if 0<=value:
                    value_list = []
                    for i in range(num_pairs):
                        value_list.append(value)
                    deg[key]= value_list
                else: 
                    raise Exception(f'{key} must be a int greater or equal to 0. Instead we got {value}')
            elif len(value) != num_pairs:
                raise Exception(f'{key} must be a list of length {num_pairs}, got {value} of length {len(value)}')      

        centers = {'center_r': center_r, 'center_c' : center_c}
        for key in centers:
            value = centers[key]
            if isinstance(value,int):
                value_list = []
                for i in range(num_pairs):
                    value_list.append(value)
                centers[key]=value_list
            elif len(value) != num_pairs:
                raise Exception(f'{key} must be a list of length {num_pairs}, got {value} of length {len(value)}')      
                        
        f_c = []
        f_r = []
        
        for i in range(num_pairs):
            fourier_r,fourier_c=gen_sparse_fourier(deg['r_pos'][i],deg['r_neg'][i],deg['c_pos'][i],
                                                   deg['c_neg'][i],deg['num_coef_c'][i],
                                                   deg['num_coef_r'][i],centers['center_r'][i],
                                                     centers['center_c'][i])
            f_c.append(fourier_c)
            f_r.append(fourier_r)
        
        return cls(f_r,f_c)

    def __evalute_torus(self,gamma : list,lamb : list,part_size : int,normalize=True, batch_size=1):
        
        result = torch.ones(size=[part_size,part_size],dtype=torch.cfloat).to(device.device)
        
        # line tensor (0/N,\dots, (N-1)/N)
        points = self._point_tensor(part_size).to(device.device)
        batches = torch.tensor_split(points,batch_size)

        cat_p = torch.empty(size=[part_size,len(batches[0])],dtype=torch.cfloat).to(device.device)
        first_p = True
        for p in batches:
            cat_q = torch.empty(size=[len(batches[0]),len(batches[0])],dtype=torch.cfloat).to(device.device)
            first_q=True
            for q in batches:
                sub_result = torch.ones(size=[p.size(0),q.size(0)],dtype=torch.cfloat).to(device.device)
                for c in range(self.n):
                    # First, we will compute the values of F and G on the whole 
                    # torus.

                    # coefficients and line indices 
                    # (-min_d+center,....,max_d+center)
                    k_F = self.k_F_line[c].to(device.device).to(torch.cfloat)
                    F = self.F[c]
                    coef_F = F['coef'].to(device.device)
                    N = coef_F.size(0)
                    
                    # Take the Kronecker product of the vectors.
                    # Kronecker product of the two vectors.
                    k_F_dot_q = torch.kron(q,k_F)
                    k_F_dot_p = torch.kron(p,k_F)
                    k_F_dot_pq = kron_sum(k_F_dot_p,k_F_dot_q)
                    
                    exp_F = torch.exp(2*torch.pi*1j*k_F_dot_pq)
                    

                    # our weights for the convolution product later, 
                    # with our coefficients
                    summer_F = torch.ones(size=[N,N],dtype = torch.cfloat).to(device.device)
                    summer_F = coef_F * summer_F

                    stride=summer_F.size(0)

                    # unsqueeze to have the correct shape
                    exp_F = exp_F.unsqueeze(0).unsqueeze(0)
                    summer_F= summer_F.unsqueeze(0).unsqueeze(0)

                    conv_F_rr = torch.conv2d(exp_F.real,summer_F.real,stride=stride).to(torch.cfloat)
                    conv_F_ii = torch.conv2d(exp_F.imag,summer_F.imag,stride=stride).to(torch.cfloat)
                    conv_F_ri = torch.conv2d(exp_F.real,summer_F.imag,stride=stride).to(torch.cfloat)
                    conv_F_ir = torch.conv2d(exp_F.imag,summer_F.real,stride=stride).to(torch.cfloat)
                    value_F = conv_F_rr - conv_F_ii + 1j*(conv_F_ri + conv_F_ir)
                    
                    value_F = value_F/(2*torch.pi)
                    value_F = value_F.squeeze(0).squeeze(0)

                    # Same thing, but for G
                    k_G = self.k_G_line[c].to(device.device).to(torch.float)
                    G = self.G[c]
                    coef_G = G['coef'].to(device.device).to(torch.float)
                    M=coef_G.size(1)
                    points = points.to(torch.float)
                    
                    p = p.to(torch.float)
                    q = q.to(torch.float)
                    # Using Kronecker products
                    k_G_dot_q = torch.kron(q,k_G)
                    k_G_dot_p = torch.kron(p,k_G)
                    k_G_dot_pq = kron_sum(k_G_dot_p,k_G_dot_q)

                    cos_G = torch.cos(2*torch.pi*k_G_dot_pq)
                    sin_G = torch.sin(2*torch.pi*k_G_dot_pq)

                    k_G = self.k_G[c].to(device.device).to(torch.float)

                    summer_G = torch.ones(size=[M,M],dtype=torch.float).to(device.device)
                            
                    summer_G_0 = coef_G[0]*summer_G
                    summer_G_1 = coef_G[1]*summer_G

                    stride = summer_G.size(0)

                    # Unsqueeze everything to have the correct amount of 
                    # dimensions
                    cos_G = cos_G.unsqueeze(0).unsqueeze(0)
                    sin_G = sin_G.unsqueeze(0).unsqueeze(0)
                    summer_G_0 = summer_G_0.unsqueeze(0).unsqueeze(0)
                    summer_G_1 = summer_G_1.unsqueeze(0).unsqueeze(0)

                    # Now to compute the values of G and its partial 
                    # derivatives on the whole space.  
                    value_G = torch.conv2d(cos_G,summer_G_0,stride=stride) + torch.conv2d(sin_G,summer_G_1,stride=stride)
                    value_G = value_G/(2*torch.pi)
                    value_G = value_G.squeeze(0).squeeze(0)

            
                    # Compute R,R_dag,delta,delta_dag to calculate 
                    # our components for our vectors.
                    # This depends on the gamma and lamb.
                    if lamb[c]<0 or lamb[c]>1:
                        raise Exception(f' lamb must be a list of values with 0 or 1.  Instead got {lamb[c]} at index {c}')
                    elif gamma[c]<0 or gamma[c]>1:
                        raise Exception(f' gamma must be a list of values with 0 or 1.  Instead got {gamma[c]} at index {c}')
                    # This is the case for R=G+sqrt(G^2+FF*) where * 
                    # is the conjugate
                    elif lamb[c]==0 and gamma[c]==0:
                        R = value_G + torch.sqrt(value_G*value_G + value_F*torch.conj_physical(value_F))
                        R = R.real
                        if normalize:
                            norm = torch.sqrt(2*R*(R-value_G))
                            R=R/norm
                        sub_result = sub_result* R
                    #This is the case for -F*
                    elif lamb[c]==0 and gamma[c]==1:
                        temp_value = -1*torch.conj_physical(value_F)
                        if normalize:
                            R_dagger = value_G - torch.sqrt(value_G*value_G + value_F*torch.conj_physical(value_F))
                            R_dagger = R_dagger.real
                            norm = torch.sqrt(2*R_dagger*(R_dagger-value_G))
                            temp_value = temp_value/norm

                        sub_result = sub_result * temp_value
                    #This is the case for F
                    elif lamb[c]==1 and gamma[c]==0:
                        temp_value = value_F
                        if normalize:
                            R=value_G + torch.sqrt(value_G*value_G + value_F*torch.conj_physical(value_F))
                            R = R.real
                            norm = torch.sqrt(2*R*(R-value_G))
                            temp_value=temp_value/norm
                        sub_result = sub_result * temp_value
                    #This is the case for R^*=G-sqrt(G^2+FF*)
                    elif lamb[c]==1 and gamma[c]==1:
                        R_dagger =value_G - torch.sqrt(value_G*value_G + value_F*torch.conj(value_F)) 
                        R_dagger = R_dagger.real
                        if normalize:
                            norm = torch.sqrt(2*R_dagger*(R_dagger-value_G))
                            R_dagger=R_dagger/norm
                        sub_result = sub_result*R_dagger
                if first_q==True:
                    cat_q = sub_result
                    first_q=False
                else:
                    cat_q = torch.cat([cat_q,sub_result],dim=0)
            if first_p == True:
                cat_p = cat_q
                first_p=False
            else:
                cat_p = torch.cat([cat_p,cat_q],dim=1)
        return cat_p

    def _evalute_torus(self,gamma : list, lamb : list,part_size : int, normalize=True):
        """ Evaluates the components of the vectors corresponding to 
        the Line bundle This is a product of all possibilities from
        the gamma and lamb to get the corresponding
        component.
        
        gamma: is a list of elements from {0,1} of length self.n 
            corresponding to 0---> no-dagger, 1--> dagger.
        lamb: is a list of elements from {0,1} of length self.n 
            corresponding to 0--> first component, 1--> second 
            component.
        part_size: is the size of the partition of our torus. 
        normalize: normalize the vectors."""
        
        # make sure the inputs are of the correct ranges
        if self.n==1 and isinstance(gamma,int):
            if 0<=gamma<=1:
                gamma = [gamma]
            else:
                raise Exception(f'gamma must be 0 or 1.  Instead we got {gamma}')
        elif self.n!=1 and isinstance(gamma,int):
            raise Exception(f'gamma must be a list of values from 0 or 1. Instead we got {gamma}')
        if self.n==1 and isinstance(lamb,int):
            if 0<=lamb<=1:
                lamb = [lamb]
            else:
                raise Exception(f'lamb must be 0 or 1.  Instead we got {lamb}')
        elif self.n!=1 and isinstance(lamb,int):
            raise Exception(f'lamb must be a list of values from 0 or 1. Instead got {lamb}')
        
        if len(gamma) != self.n:
            raise Exception(f'gamma must be a list of length {self.n}, got {gamma} of length {len(gamma)}')

        if len(lamb) != self.n:
            raise Exception(f'lamb must be a list of length {self.n}, got {lamb} of length {len(lamb)}')
                
        if part_size<=0:
            raise Exception(f'part_size must be a strictly positive integer.  Instead we got {part_size}')
        
        batch_size = 1
        while True:
            try: 
                return self.__evalute_torus(gamma, lamb,part_size,normalize, batch_size)
            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.'):
                    batch_size+=1
                    continue
                else:
                    raise e

    def __evalute_normal(self,gamma : list, part_size : int, batch_size=1):
        result = torch.ones(size=[part_size,part_size],dtype=torch.cfloat).to(device.device)
        
        # line tensor (0/N,\dots, (N-1)/N)
        points = self._point_tensor(part_size).to(device.device)
        batches = torch.tensor_split(points,batch_size)
        for p in batches:
            for q in batches:
                for c in range(self.n):
                    # First, we will compute the values of F and G on the whole 
                    # torus.

                    # coefficients and line indices 
                    # (-min_d+center,....,max_d+center)
                    k_F = self.k_F_line[c].to(device.device).to(torch.cfloat)
                    F = self.F[c]
                    coef_F = F['coef'].to(device.device)
                    N = coef_F.size(0)
                    
                    # Take the Kronecker product of the vectors.
                    # Kronecker product of the two vectors.
                    k_F_dot_q = torch.kron(q,k_F)
                    k_F_dot_p = torch.kron(p,k_F)
                    k_F_dot_pq = kron_sum(k_F_dot_p,k_F_dot_q)
                    
                    exp_F = torch.exp(2*torch.pi*1j*k_F_dot_pq)
                    

                    # our weights for the convolution product later, 
                    # with our coefficients
                    summer_F = torch.ones(size=[N,N],dtype = torch.cfloat).to(device.device)
                    summer_F = coef_F * summer_F

                    stride=summer_F.size(0)

                    # unsqueeze to have the correct shape
                    exp_F = exp_F.unsqueeze(0).unsqueeze(0)
                    summer_F= summer_F.unsqueeze(0).unsqueeze(0)

                    conv_F_rr = torch.conv2d(exp_F.real,summer_F.real,stride=stride).to(torch.cfloat)
                    conv_F_ii = torch.conv2d(exp_F.imag,summer_F.imag,stride=stride).to(torch.cfloat)
                    conv_F_ri = torch.conv2d(exp_F.real,summer_F.imag,stride=stride).to(torch.cfloat)
                    conv_F_ir = torch.conv2d(exp_F.imag,summer_F.real,stride=stride).to(torch.cfloat)
                    value_F = conv_F_rr - conv_F_ii + 1j*(conv_F_ri + conv_F_ir)
                    
                    value_F = value_F/(2*torch.pi)
                    value_F = value_F.squeeze(0).squeeze(0)

                    # Same thing, but for G
                    k_G = self.k_G_line[c].to(device.device).to(torch.float)
                    G = self.G[c]
                    coef_G = G['coef'].to(device.device).to(torch.float)
                    M=coef_G.size(1)
                    points = points.to(torch.float)
                    
                    p = p.to(torch.float)
                    q = q.to(torch.float)
                    # Using Kronecker products
                    k_G_dot_q = torch.kron(q,k_G)
                    k_G_dot_p = torch.kron(p,k_G)
                    k_G_dot_pq = kron_sum(k_G_dot_p,k_G_dot_q)

                    cos_G = torch.cos(2*torch.pi*k_G_dot_pq)
                    sin_G = torch.sin(2*torch.pi*k_G_dot_pq)

                    k_G = self.k_G[c].to(device.device).to(torch.float)

                    summer_G = torch.ones(size=[M,M],dtype=torch.float).to(device.device)
                            
                    summer_G_0 = coef_G[0]*summer_G
                    summer_G_1 = coef_G[1]*summer_G

                    stride = summer_G.size(0)

                    # Unsqueeze everything to have the correct amount of 
                    # dimensions
                    cos_G = cos_G.unsqueeze(0).unsqueeze(0)
                    sin_G = sin_G.unsqueeze(0).unsqueeze(0)
                    summer_G_0 = summer_G_0.unsqueeze(0).unsqueeze(0)
                    summer_G_1 = summer_G_1.unsqueeze(0).unsqueeze(0)

                    # Now to compute the values of G and its partial 
                    # derivatives on the whole space.  
                    value_G = torch.conv2d(cos_G,summer_G_0,stride=stride) + torch.conv2d(sin_G,summer_G_1,stride=stride)
                    value_G = value_G/(2*torch.pi)
                    value_G = value_G.squeeze(0).squeeze(0)

            
                    # Compute R,R_dag,delta,delta_dag to calculate 
                    # our components for our vectors.
                    # This depends on the gamma and lamb.
                    
                    if gamma[c]<0 or gamma[c]>1:
                        raise Exception(f' gamma must be a list of values with 0 or 1.  Instead got {gamma[c]} at index {c}')
                   
                    elif gamma[c]==0:
                        R = value_G + torch.sqrt(value_G*value_G + value_F*torch.conj_physical(value_F))
                        R = R.real
                        return torch.sqrt(2*R*(R-value_G))  
                    
                    elif gamma[c]==1:
                        R_dagger =value_G - torch.sqrt(value_G*value_G + value_F*torch.conj(value_F)) 
                        R_dagger = R_dagger.real
                        
                        return torch.sqrt(2*R_dagger*(R_dagger-value_G))

    def _evalute_normal(self,gamma : list, part_size : int):
        # make sure the inputs are of the correct ranges
        if self.n==1 and isinstance(gamma,int):
            if 0<=gamma<=1:
                gamma = [gamma]
            else:
                raise Exception(f'gamma must be 0 or 1.  Instead we got {gamma}')
        elif self.n!=1 and isinstance(gamma,int):
            raise Exception(f'gamma must be a list of values from 0 or 1. Instead we got {gamma}')
        
        if len(gamma) != self.n:
            raise Exception(f'gamma must be a list of length {self.n}, got {gamma} of length {len(gamma)}')
         
        if part_size<=0:
            raise Exception(f'part_size must be a strictly positive integer.  Instead we got {part_size}')
        
        batch_size = 1
        while True:
            try: 
                return self.__evalute_normal(gamma,part_size, batch_size)
            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.'):
                    batch_size+=1
                    continue
                else:
                    raise e

    def output_image(self,gamma : list,lamb : list,part_size : int,normalize=True):
        """Outputs a matplotlib image of a component with the 
        real and imaginary part plotted.
        
        gamma: is a list of elements from {0,1} of length self.n 
            corresponding to 0---> no-dagger, 1--> dagger.
        lamb: is a list of elements from {0,1} of length self.n 
            corresponding to 0--> first component, 1--> second 
            component.
        part_size: is the size of the partition of our torus For
            example, N=part_size gives us a partition of the torus
            into N^2 squares of dimension 1/Nx1/N. This gives us the 
            resolution of the image as well. 
        normalize: normalize the vectors."""



        values = self._evalute_torus(gamma,lamb,part_size,normalize).to('cpu')
        real = values.real
        imag = values.imag


        fig = plt.figure(figsize=(8,8))
        fig.add_subplot(2,2,1)
        plt.title(f'real: gamma={gamma},\n lambda={lamb}')
        plt.axis('off')
        real_image = plt.imshow(real,cmap='RdBu')
        
        

        fig.add_subplot(2,2,3)
        plt.title(f'Imag: gamma={gamma},\n lambda={lamb}')
        plt.axis('off')
        imaginary_image = plt.imshow(imag,cmap='RdBu')
        
        fig.add_subplot(1,3,3)
        plt.axis('off')
        plt.colorbar()
        
        
        plt.show()
   
    def output_Fourier_poly(self,index: int):
        """Prints the real and complex Fourier polynomials
        for one of the Haldane pairs.
        
        index: the Haldane pair in the Haldane sequence"""

        if index<0:
            raise Exception(f'index must be a positive integer. Instead we got {index}')

        F=self.F[index]
        G=self.G[index]
        Max_1_F = F['coef'].size(0)
        Max_2_F = F['coef'].size(1)
        text_F = "F="
        for k_1 in range(Max_1_F):
            for k_2 in range(Max_2_F):
                if F['coef'][k_1][k_2]!=0:
                    text_F += f"{F['coef'][k_1][k_2]}exp(2pi i {(k_1-F['min_d']+F['center'],k_2-F['min_d']+F['center'])}* p) +\n "
        print(text_F)

        Max_1_G = G['coef'].size(1)
        Max_2_G = G['coef'].size(2)
        text_G = "G="
        for k_1 in range(Max_1_G):
            for k_2 in range(Max_2_G):
                if G['coef'][0][k_1][k_2]!=0:
                    text_G += f"{G['coef'][0][k_1][k_2]} Cos(2pi {(k_1-G['min_d']+G['center'],k_2-G['min_d']+G['center'])}* p)+\n"
                if G['coef'][1][k_1][k_2]!=0:
                    text_G += f" {G['coef'][1][k_1][k_2]} Sin(2pi({(k_1-G['min_d']+G['center'],k_2-G['min_d']+G['center'])}*p)) +\n "
        print(text_G)

    def output_rep(self,part_size : int, normalize = True):
        """Returns a tensor of shape [2^n,2^n,N,N] where n=number of tensor 
        products, N= is the partition size, corresponding to all the 
        components of the vectors.
        
        part_size: is the size of the partition of our torus For
            example, N=part_size gives us a partition of the torus
            into N^2 squares of dimensions 1/Nx1/N.
            
            """

        if part_size<=0:
            raise Exception(f'part_size must be a strictly positive integer. Instead we got {part_size}')
        
        # We need a list of all possible combintations of gamma and lamb, 
        # since these give us our components. 
        gamma_list = gen_binary_list(self.n,list())
        lambda_list = gamma_list 
        #Since they are the same 
        
        result = torch.zeros(size=[np.power(2,self.n),np.power(2,self.n),part_size,part_size])
        result = result.type(torch.complex64).to(device.device)
        # The size here is: 2^n number of possibilties for gamma, 2^n possibilties for lamb, 
        # and [part_size, part_size] for the resolution of the torus. 

        len_gl = len(gamma_list)
        for i in range(len_gl):
            for j in range(len_gl):
                result[i][j] = self._evalute_torus(gamma_list[i],lambda_list[j],part_size,normalize=normalize)

        return result.to('cpu')

    def output_normal(self,part_size : int):
        if part_size<=0:
            raise Exception(f'part_size must be a strictly positive integer. Instead we got {part_size}')
        
        # We need a list of all possible combintations of gamma and lamb, 
        # since these give us our components. 
        gamma_list = gen_binary_list(self.n,list())

        result = torch.zeros(size=[len(gamma_list),part_size,part_size])
        result = result.type(torch.complex64).to(device.device)
        # The size here is: 2^n number of possibilties for gamma, 2^n possibilties for lamb, 
        # and [part_size, part_size] for the resolution of the torus. 

        len_gl = len(gamma_list)
        for i in range(len_gl):
            result[i] = self._evalute_normal(gamma_list[i],part_size)

        return result.to('cpu')

    def _evalute_curvature(self,part_size: int,batch_size: int):
        """Approximates the Curvature form on each patch of the torus,
        depedent on the size of the partition of the torus. See the 
        document for the proof and the equations used here.  
        Essentially, since our line bundles is just a tensor 
        product of Haldane bundles, and the Chern number of a 
        tensor product is just the sum of the Chern numbers for 
        each piece.  Then we can calculate each piece to determine 
        our chern number.This heavily uses the GPU to do the 
        calculations.
        
        part_size: is the size of the partition of our torus For
            example, N=part_size gives us a partition of the torus
            into N^2 squares of dimension 1/Nx1/N.
        batch_size: batches the torus"""
        
        result = 0
        
        #Batch the torus into smaller chunks for better use of the GPU    
        points = self._point_tensor(part_size).to(device.device)
        batches = torch.tensor_split(points,batch_size)
        
        del points
        

        for p in batches:
            for q in batches:
                for c in range(self.n):
                    
                    # Get coefficients and tensors (-min_d+center,....,max_d+center)
                    k_F = self.k_F_line[c].to(device.device).to(torch.cfloat)
                    F = self.F[c]
                    coef_F = F['coef'].to(device.device)
                    N = coef_F.size(0)

                    # Kronecker product of the two vectors.
                    k_F_dot_q = torch.kron(q,k_F)
                    k_F_dot_p = torch.kron(p,k_F)
                    k_F_dot_pq = kron_sum(k_F_dot_p,k_F_dot_q)
                    
                    exp_F = torch.exp(2*torch.pi*1j*k_F_dot_pq)
                    exp_F_conj = torch.exp(-2*torch.pi*1j*k_F_dot_pq)

                    del k_F_dot_q
                    del k_F_dot_p
                    del k_F_dot_pq
                    

                    # coefficients for the partial derivatives
                    k_F = self.k_F[c].to(device.device)
                    k_F_x = k_F[:,:,0]
                    k_F_y = k_F[:,:,1]

                   

                    # Weights for the convlution product later, with the coefficients
                    # for F, F_x, F_y, F*_x, F*_y.
                    summer_F = torch.ones(size=[N,N],dtype = torch.cfloat).to(device.device)
                    summer_F = coef_F * summer_F
                    summer_F_x = 1j*k_F_x * summer_F
                    summer_F_y =1j* k_F_y * summer_F
                    summer_conjF_x = -1j*k_F_x * torch.conj_physical(summer_F)
                    summer_conjF_y = -1j*k_F_y * torch.conj_physical(summer_F)

                    

                    del coef_F
                    del k_F_x
                    del k_F_y
                    del k_F
                    torch.cuda.empty_cache()
                    
                    stride=summer_F.size(0)

                    exp_F = exp_F.unsqueeze(0).unsqueeze(0)
                    exp_F_conj=exp_F_conj.unsqueeze(0).unsqueeze(0)
                    summer_F= summer_F.unsqueeze(0).unsqueeze(0)
                    summer_F_x = summer_F_x.unsqueeze(0).unsqueeze(0)
                    summer_F_y = summer_F_y.unsqueeze(0).unsqueeze(0)
                    summer_conjF_x=summer_conjF_x.unsqueeze(0).unsqueeze(0)
                    summer_conjF_y=summer_conjF_y.unsqueeze(0).unsqueeze(0)

                    
                    # Convlution product, since these are complex-valued
                    # we needed to seperate everything out into real 
                    # and imaginary parts to apply the convolution methods
                    conv_F_rr = torch.conv2d(exp_F.real,summer_F.real,stride=stride).to(torch.cfloat)
                    conv_F_ii = torch.conv2d(exp_F.imag,summer_F.imag,stride=stride).to(torch.cfloat)
                    conv_F_ri = torch.conv2d(exp_F.real,summer_F.imag,stride=stride).to(torch.cfloat)
                    conv_F_ir = torch.conv2d(exp_F.imag,summer_F.real,stride=stride).to(torch.cfloat)
                    value_F = conv_F_rr - conv_F_ii + 1j*(conv_F_ri + conv_F_ir) 
                    
                    value_F = value_F/(2*torch.pi)

                    conv_F_x_rr = torch.conv2d(exp_F.real,summer_F_x.real,stride=stride).to(torch.cfloat)
                    conv_F_x_ii = torch.conv2d(exp_F.imag,summer_F_x.imag,stride=stride).to(torch.cfloat)
                    conv_F_x_ri = torch.conv2d(exp_F.real,summer_F_x.imag,stride=stride).to(torch.cfloat)
                    conv_F_x_ir = torch.conv2d(exp_F.imag,summer_F_x.real,stride=stride).to(torch.cfloat)
                    F_x = conv_F_x_rr - conv_F_x_ii + 1j*(conv_F_x_ri + conv_F_x_ir)
                    
                    conv_F_y_rr = torch.conv2d(exp_F.real,summer_F_y.real,stride=stride).to(torch.cfloat)
                    conv_F_y_ii = torch.conv2d(exp_F.imag,summer_F_y.imag,stride=stride).to(torch.cfloat)
                    conv_F_y_ri = torch.conv2d(exp_F.real,summer_F_y.imag,stride=stride).to(torch.cfloat)
                    conv_F_y_ir = torch.conv2d(exp_F.imag,summer_F_y.real,stride=stride).to(torch.cfloat)
                    F_y = conv_F_y_rr - conv_F_y_ii + 1j*(conv_F_y_ri + conv_F_y_ir)

                    conv_conjF_x_rr = torch.conv2d(exp_F_conj.real,summer_conjF_x.real,stride=stride).to(torch.cfloat)
                    conv_conjF_x_ii = torch.conv2d(exp_F_conj.imag,summer_conjF_x.imag,stride=stride).to(torch.cfloat)
                    conv_conjF_x_ri = torch.conv2d(exp_F_conj.real,summer_conjF_x.imag,stride=stride).to(torch.cfloat)
                    conv_conjF_x_ir = torch.conv2d(exp_F_conj.imag,summer_conjF_x.real,stride=stride).to(torch.cfloat)
                    conjF_x = conv_conjF_x_rr - conv_conjF_x_ii + 1j*(conv_conjF_x_ri + conv_conjF_x_ir)
                    
                    conv_conjF_y_rr = torch.conv2d(exp_F_conj.real,summer_conjF_y.real,stride=stride).to(torch.cfloat)
                    conv_conjF_y_ii = torch.conv2d(exp_F_conj.imag,summer_conjF_y.imag,stride=stride).to(torch.cfloat)
                    conv_conjF_y_ri = torch.conv2d(exp_F_conj.real,summer_conjF_y.imag,stride=stride).to(torch.cfloat)
                    conv_conjF_y_ir = torch.conv2d(exp_F_conj.imag,summer_conjF_y.real,stride=stride).to(torch.cfloat)
                    conjF_y = conv_conjF_y_rr - conv_conjF_y_ii + 1j*(conv_conjF_y_ri + conv_conjF_y_ir)
                    

                    
                    
                    del exp_F
                    del exp_F_conj
                    del summer_F
                    del summer_F_x
                    del summer_F_y
                    del summer_conjF_x
                    del summer_conjF_y
                    
                    # For G
                    k_G = self.k_G_line[c].to(device.device).to(torch.float)
                    G = self.G[c]
                    coef_G = G['coef'].to(device.device).to(torch.float)
                    M=coef_G.size(1)

                    p = p.to(torch.float)
                    q = q.to(torch.float)
                    # Using Kronecker products
                    k_G_dot_q = torch.kron(q,k_G)
                    k_G_dot_p = torch.kron(p,k_G)
                    k_G_dot_pq = kron_sum(k_G_dot_p,k_G_dot_q)

                    cos_G = torch.cos(2*torch.pi*k_G_dot_pq)
                    sin_G = torch.sin(2*torch.pi*k_G_dot_pq)

                    del k_G_dot_p
                    del k_G_dot_q
                    del k_G_dot_pq
                    torch.cuda.empty_cache()

                    k_G = self.k_G[c].to(device.device).to(torch.float)
                    
                    k_G_x = k_G[:,:,0]
                    k_G_y = k_G[:,:,1]
                    
                    # Get the coefficients
                    summer_G = torch.ones(size=[M,M],dtype=torch.float).to(device.device)
                    
                    summer_G_0 = coef_G[0]*summer_G
                    summer_G_1 = coef_G[1]*summer_G
                    summer_G_0_x = k_G_x * summer_G_0
                    summer_G_0_y = k_G_y * summer_G_0
                    summer_G_1_x = k_G_x * summer_G_1
                    summer_G_1_y = k_G_y * summer_G_1

                    del coef_G
                    del k_G
                    del k_G_x
                    del k_G_y
                    torch.cuda.empty_cache()

                    stride = summer_G.size(0)

                    # Unsqueeze everything to have the correct amount of dimensions
                    cos_G = cos_G.unsqueeze(0).unsqueeze(0)
                    sin_G = sin_G.unsqueeze(0).unsqueeze(0)
                    summer_G_0 = summer_G_0.unsqueeze(0).unsqueeze(0)
                    summer_G_1 = summer_G_1.unsqueeze(0).unsqueeze(0)
                    summer_G_0_x = summer_G_0_x.unsqueeze(0).unsqueeze(0)
                    summer_G_0_y = summer_G_0_y.unsqueeze(0).unsqueeze(0)
                    summer_G_1_x = summer_G_1_x.unsqueeze(0).unsqueeze(0)
                    summer_G_1_y = summer_G_1_y.unsqueeze(0).unsqueeze(0)

                    # Now to compute the values of G and its partial derivatives on the whole space.  
                    value_G = torch.conv2d(cos_G,summer_G_0,stride=stride) + torch.conv2d(sin_G,summer_G_1,stride=stride)
                    value_G = value_G/(2*torch.pi)
                    
                    G_x = torch.conv2d(-1*sin_G,summer_G_0_x,stride=stride) + torch.conv2d(cos_G,summer_G_1_x,stride=stride)
                    G_y = torch.conv2d(-1*sin_G,summer_G_0_y,stride=stride) + torch.conv2d(cos_G,summer_G_1_y,stride=stride)

                    del cos_G
                    del sin_G
                    del summer_G_0
                    del summer_G_1
                    del summer_G_0_x
                    del summer_G_1_x
                    del summer_G_0_y
                    del summer_G_1_y

                    # Next, we will need to defined indicator function 
                    # that measures which points on the torus have
                    # G(p)>0 or G(p)<0, which determines which
                    # curvature form we use. 
                    V = torch.relu(value_G.clone())
                    V[V!=0]=1
                    V_dag = torch.relu(-value_G.clone())
                    V_dag[V_dag!=0]=1
                    


                    # Now we can start computing our curvature.  We will use V* Omega + V_dag * Omega_dag to
                    # do our computation and to make it where we get the computations in the right spot with
                    # the correct neighborhood.

                    
                    quasi_norm = torch.sqrt(value_G*value_G + value_F*torch.conj_physical(value_F))
                    quasi_norm = quasi_norm.real

                    # Get our values for R,delta
                    R = value_G + quasi_norm
                    div_delta =torch.sqrt(2*R*(R-value_G))
                    delta = 1/div_delta

                    #Compute the partial derivatives of R,delta.
                    R_x = G_x + (value_G*G_x + (1/2)*value_F*conjF_x + (1/2)*F_x*torch.conj_physical(value_F))/(quasi_norm)
                    R_x = torch.real(R_x)
                    R_y = G_y + (value_G*G_y + (1/2)*value_F*conjF_y + (1/2)*F_y*torch.conj_physical(value_F))/(quasi_norm)
                    R_y = torch.real(R_y)

                    delta_x = (-2*R*R_x + R_x*value_G + G_x*R)/ torch.pow(div_delta,3)
                    delta_y =  (-2*R*R_y + R_y*value_G + G_y*R)/ torch.pow(div_delta,3)
                    
                    del R_x
                    del R_y
                    del R
                    torch.cuda.empty_cache()

                    #Compute the curvature in this neighborhood
                    Psi = delta*torch.conj_physical(value_F)*(delta_x*F_y - delta_y*F_x)
                    Psi += delta*value_F*(conjF_x*delta_y - conjF_y*delta_x)
                    Psi += delta*delta*(conjF_x*F_y - conjF_y*F_x)       
                    Psi = torch.real(1j*Psi/(2*torch.pi))    


                    # Get our values for R_dagger, delta_dagger
                    R_dagger = value_G - quasi_norm
                    div_delta_dagger = torch.sqrt(2*R_dagger*(R_dagger-value_G))
                    delta_dagger = 1/div_delta_dagger

                    #Compute the partial derivatives of R_dagger and delta_dagger
                    R_dagger_x = G_x - (value_G*G_x + (1/2)*value_F*conjF_x + (1/2)*F_x*torch.conj_physical(value_F))/(quasi_norm)
                    R_dagger_x = torch.real(R_dagger_x)
                    R_dagger_y = G_y - (value_G*G_y + (1/2)*value_F*conjF_y + (1/2)*F_y*torch.conj_physical(value_F))/(quasi_norm)  
                    R_dagger_y = torch.real(R_dagger_y)

                    delta_dag_x = (-2*R_dagger*R_dagger_x + R_dagger_x*value_G + G_x*R_dagger)/ torch.pow(div_delta_dagger,3)
                    delta_dag_y = (-2*R_dagger*R_dagger_y + R_dagger_y*value_G + G_y*R_dagger)/ torch.pow(div_delta_dagger,3)

                    del R_dagger
                    del R_dagger_x
                    del R_dagger_y
                    torch.cuda.empty_cache()
                    # Compute the curvature in this neighborhood
                    Psi_dag = delta_dagger*value_F*(delta_dag_x*conjF_y - delta_dag_y*conjF_x)
                    Psi_dag += delta_dagger*torch.conj_physical(value_F)*(F_x*delta_dag_y - F_y*delta_dag_x)
                    Psi_dag +=delta_dagger*delta_dagger*(F_x*conjF_y - F_y*conjF_x)
                    Psi_dag = torch.real(1j*Psi_dag/(2*torch.pi))  

                    # Check if there is any nan that appear and make them zero
                    # We need to do this since we computed Psi and Psi_dag on 
                    # the whole torus, and there might be some points
                    # where they are not defined.
                    Psi[torch.isnan(Psi)]=0
                    Psi_dag[torch.isnan(Psi_dag)]=0
                    
                    #add the curvature forms with the indicator function
                    sub_result = V * Psi + V_dag * Psi_dag 
                    result += torch.sum(sub_result).to('cpu')
        return int(np.round(result/(np.power(part_size,2))))
    
    def Chern_number(self,part_size : int,batch_size : int):
        """Approximates the Chern number.
        Note that this will attempt to calculate the Chern number,
        with the inputed parameters, and increase the batch_size
        if needed,.
        part_size: is the size of the partition of our torus For
            example, N=part_size gives us a partition of the torus
            into N^2 squares of dimension 1/Nx1/N.
        batch_size: batch the torus"""
        if part_size<=0:
            raise Exception(f'lat_size must be a strictly positive integer. The value of part_size was {part_size}')
        if batch_size<=0:
            raise Exception(f'batch_size must be a strictly positive integer.  The value of batch_size was {batch_size}')
        while True:
            try: 
                return self._evalute_curvature(part_size,batch_size)
            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.'):
                    batch_size+=1
                    continue
                else:
                    raise e
            
    def _k_tensor_box(self):
        """Integer Tensor of shape [N,N,2] of indices (k_1,k_2)."""
        result_F = []
        result_G = []
        for a in range(self.n):
            # First, construct the k-tensor for F
            N= self.F[a]['coef'].size(0)
            sub_F = torch.zeros(size=[N,N,2],dtype=torch.int)

            for i in range(N):
                for j in range(N):
                    sub_F[i][j] = torch.tensor([i-self.F[a]['min_d']+ self.F[a]['center'],j-self.F[a]['min_d'] + self.F[a]['center']])
            result_F.append(sub_F)
            
            M = self.G[a]['coef'].size(1)
            sub_G = torch.zeros(size=[M,M,2],dtype=torch.int)

            for i in range(M):
                for j in range(M):
                    sub_G[i][j] = torch.tensor([i - self.G[a]['min_d'] + self.G[a]['center'],j - self.G[a]['min_d'] + self.G[a]['center']])
            result_G.append(sub_G)
        return result_G, result_F

    def _k_tensor_line(self):
        """ Generates a integer tensor of shape [N]
        with entires (-min_d+center,..,-min_d,.., max_d+center)."""

        result_F = []
        result_G = []
        for a in range(self.n):
            N=self.F[a]['coef'].size(0)
            sub_F = torch.zeros(size=[N],dtype=torch.int)
            
            for i in range(N):
                sub_F[i] = torch.tensor([i-self.F[a]['min_d'] + self.F[a]['center']])
            result_F.append(sub_F)
            M=self.G[a]['coef'].size(1)
            sub_G = torch.zeros(size=[M],dtype = torch.int)

            for i in range(M):
                sub_G[i] = torch.tensor([i-self.G[a]['min_d'] + self.G[a]['center']])
            result_G.append(sub_G)
        return result_G,result_F
    
    
    def _point_tensor(self,part_size: int):
        """Generates a tensor of shape [N], with entires 
        (0/N,...,(i-1)/N,...,(N-1)/N)"""

        if part_size<=0:
            raise Exception(f'part_size must be strictly positive. Instead we got: {part_size}')
        
        result = torch.zeros(size=[part_size],dtype=torch.float)
        for i in range(part_size):
            result[i] = i/part_size
        return result


#These are our important non-methods for the Fourier_Bundle class

# This function generates a list of (a_1,\dots, a_n) with a_i\in Z_2.
def gen_binary_list(size: int,input: list):
    """Generates a list of (a_1,...,a_n) with a_i\in Z_2,
    recursivly
    
    size: the length of the list n."""

    if size<=0:
        raise Exception(f'The value of size needs to be a strictly positive integer.  Instead we got : {size}')
    result = []
    if input==None:
        input = []
    if len(input)==size:
        return [input]
    
    #add 0 to the list
    input_1 = input.copy()
    input_1.append(0)
    result_1=gen_binary_list(size,input_1)

    #add 1 to the list
    input_2 = input.copy()
    input_2.append(1)
    result_2=gen_binary_list(size,input_2)

    result.extend(result_1)
    result.extend(result_2)
    return result



def gen_fourier(r_pos: int,r_neg: int,c_pos : int,c_neg: int, center_r : int, center_c : int):
    """Generates a random Haldane pair (G,F) of Fourier polynomials.

    r_pos: positive integer for real-valued Fourier polynomial
        representing max positive degree from (0,0).
    r_neg: positive integer for the real-valued Fourier
        polynomial representing max negative degree from (0,0).
    c_pos: positive integer for complex-valued Fourier
        polynomial representing max positive degree from (0,0).
    c_neg: positive integer for complex-valued Fourier
        polynmial representing max negative degree from (0,0).
    center_r: (default = 0) integer for real-valued Fourier 
        polynomial representing the shift from the index (0,0).
    center_c: (default = 0) integer for complex-valued Fourier 
        polynomial representing the shift from the index (0,0).
    """

    if r_pos<0:
        raise Exception(f'r_pos should be a positive integer. Instead we got: {r_pos}')
    if r_neg<0:
        raise Exception(f'r_neg should be a positive integer. Instead we got: {r_neg}')
    if c_pos<0:
        raise Exception(f'c_pos should be a positive integer. Instead we got: {c_pos}')
    if c_neg<0:
        raise Exception(f'c_neg should be a positive integer. Instead we got: {c_neg}')
    
    #generate the complex coefficients for F
    size_c=[c_pos+c_neg+1,c_pos+c_neg+1]
    coef_c = torch.rand(size=size_c,dtype=torch.cfloat).uniform_(-2,2)

    F= {'coef': coef_c, 'min_d':c_neg, 'center' : center_c}
    #generate the real coefficients for G

    size_r = [2,r_pos+r_neg+1,r_pos+r_neg+1]
    coef_r = torch.rand(size=size_r,dtype=torch.float).uniform_(-2,2)

    G={'coef': coef_r, 'min_d':r_neg, 'center' : center_r}

    return G,F


def gen_sparse_fourier(r_pos: int,r_neg: int,c_pos: int,c_neg: int,num_coef_c: int,
                       num_coef_r: int, center_r : int, center_c : int):
    """ Generates random Haldan pair (G,F) of Fourier polynomials
    with the coefficients sparsely chosen.
    
    r_pos: list of positive integers for real-valued Fourier polynomials
        representing max positive degree from (0,0).
    r_neg: list of positive integers for the real-valued Fourier
        polynomials representing max negative degree from (0,0).
    c_pos: list of positive integers for complex-valued Fourier
        polynomials representing max positive degree from (0,0).
    c_neg: list of positive integers for complex-valued Fourier
        polynmials representing max negative degree from (0,0).
    num_coef_c: max number of non-zero coefficients for F
    num_coef_r: max number of non-zero coefficients for G
    center_r: (default = 0) list of integers for real-valued Fourier 
        polynomials representing the shift from the index (0,0).
    center_c: (default = 0) list of integers for complex-valued Fourier 
        polynomials representing the shift from the index (0,0).
    """

    N = c_pos+c_neg+1
    size_c = [N,N]
    coef_c = torch.zeros(size=size_c,dtype=torch.cfloat)
    
    rand_index = np.random.randint(0,N,size=[num_coef_c,2])
    for p in rand_index:
        coef_c[p[0]][p[1]] = torch.rand(size=[1],dtype=torch.cfloat).uniform_(-2,2)
    
    F = {'coef': coef_c, 'min_d': c_neg, 'center' : center_c}

    #generate the real coefficeints for G sparsely
    M = r_pos + r_neg + 1
    size_r = [2,M,M]
    coef_r = torch.zeros(size=size_r,dtype=torch.float)

    rand_index = np.random.randint(0,M,size=[num_coef_r,2])
    for p in rand_index:
        cos_sin = np.random.randint(0,2)
        coef_r[cos_sin][p[0]][p[1]] = torch.rand(size=[1],dtype=torch.float).uniform_(-2,2)
    
    G = {'coef': coef_r, 'min_d': r_neg, 'center' : center_r}
    return G,F


def kron_sum(input,other):
    """Kronecker sum of two vectors, order matters.

    input: left input vector,
    other: right input vector"""
    m=input.size(0)
    n=other.size(0)
    stack_in = []
    stack_ot = []
    for i in range(n):
        stack_in.append(input)

    for j in range(m):
        stack_ot.append(other)
    
    input_stack = torch.stack(stack_in)
    other_stack = torch.stack(stack_ot)
    return input_stack + other_stack.T







        


    
                
