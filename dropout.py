'''
Name: Santosh Kumar Kasam
M no: 1533833
'''
import torch
import torch.nn as nn


class Dropout(nn.Module):
    
    def __init__(self, p=0.1):
        super().__init__()
        # store p
        self.p = p
        
    def forward(self, x):
        # In training mode, set each value 
        # independently to 0 with probability p
        # and scale the remaining values 
        # according to the lecture
        if self.training: 
            
            #generating a matrix with dimensions of x with uniform random distribution
            d_rand = torch.rand_like(x)
            
            #if a value in the above matrix is greater than p value, it changes to 1(true) else 0(false)
            d_oneandzeros = d_rand > self.p
            
            #multiplying the input matrix with above matrix zeros the dropout probability amount of nodes
            #and dividing with the value of (1-p) upscales the magnitudes of the nodes
            x_do = (x*d_oneandzeros)/(1-self.p)
            
            return x_do
                        
        # In evaluation mode, return the
        # unmodified input
        else:            
            return x
            
            
      