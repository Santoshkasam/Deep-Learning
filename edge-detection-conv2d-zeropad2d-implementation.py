'''
Name : Santosh Kumar Kasam
M no: 153333
Assignment: 7
'''
import torch
import torch.nn as nn


# modify the edge detector kernel in such a way that
# it calculates the derivatives in x and y direction
edge_detector_kernel = torch.tensor([[[[-1.0, 0.0],
          [1.0, 0.0]]],

        [[[-1.0, 1.0],
          [0.0, 0.0]]]])

class Conv2d(nn.Module):
    
    def __init__(self, kernel, padding=0, stride=1):
        super().__init__()
        self.kernel = nn.Parameter(kernel)
        self.padding = ZeroPad2d(padding)
        self.stride = stride
        
    def forward(self, x):
        x = self.padding(x)
        # For input of shape C x H x W
        # implement the convolution of x with self.kernel
        # using self.stride as stride
        # The output is expected to be of size C x H' x W'
        ker = self.kernel 
        ker_h = ker.shape[2]
        ker_w = ker.shape[3]
        s = self.stride
        
        # determining the shape of the output and intializing the output tensor with zeros
        outputc_shape = ker.shape[0]
        outputh_shape = int( ( x.shape[1] - ker.shape[2] ) / s ) + 1             
        outputw_shape = int( ( x.shape[2] - ker.shape[3] ) / s ) + 1        
        output = torch.zeros(outputc_shape, outputh_shape, outputw_shape)
        
        # updating the output tensor values with convoluted results
        for c in range(outputc_shape):
            for h_out in range(outputh_shape):
                for w_out in range(outputw_shape):                    
                    output[c,h_out,w_out] = torch.sum(x[0:x.shape[0],s*h_out:s*h_out+ker_h, s*w_out:s*w_out+ker_w] * ker[c])  
                    
        return output


class ZeroPad2d(nn.Module):
    
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        
    def forward(self, x):
        # For input of shape C x H x W
        # return tensor zero padded equally at left, right,
        # top, bottom such that the output is of size
        # C x (H + 2 * self.padding) x (W + 2 * self.padding)
        pad = self.padding
        
        # preparing pads for sides, top and bottom
        side_pad = torch.zeros(x.shape[0],x.shape[1],pad)
        top_bot_pad = torch.zeros(x.shape[0],pad,2*pad+x.shape[2])
        
        # concatenating pads to the input tensor
        x_sidepadding = torch.cat((side_pad,x,side_pad), 2)       
        x_padded = torch.cat((top_bot_pad,x_sidepadding,top_bot_pad),1)        
        
        # returning the padded tensor
        return x_padded
