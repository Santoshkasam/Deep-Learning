import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # kaiming he initialization
        fan_in = m.in_features
        fan_out = m.out_features
        sd = (2/fan_in)**0.5  
        m.weight.data = torch.randn(fan_out,fan_in) * sd
       

class BatchNorm(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        # set theta_mu and theta_sigma such that the output of
        # forward initially is zero centered and 
        # normalized to variance 1
        theta_mu = torch.zeros(num_channels)
        theta_sigma = torch.ones(num_channels)
        self.theta_mu = nn.Parameter(theta_mu)
        self.theta_sigma = nn.Parameter(theta_sigma)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        if self.training:
            # specify behavior at training time
            stat_mean = x.mean(axis=0)
            stat_var = x.var(axis=0)
            if self.running_mean is None:
                # set the running stats to stats of x
                self.running_mean = stat_mean
                self.running_var = stat_var               
            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x
                self.running_mean = 0.9 * self.running_mean + 0.1 * stat_mean
                self.running_var = 0.9 * self.running_var + 0.1 * stat_var
            x1=(x- stat_mean) /(stat_var + self.eps)**0.5
            ret =   self.theta_mu + self.theta_sigma * x1
            return ret
               
        else:
            stat_mean = x.mean(axis=0)
            stat_var = x.var(axis=0)
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                normalized_x = (x - stat_mean) / ( stat_var + self.eps)**(0.5)
            else:
                # use running stats for normalization
                normalized_x = (x - self.running_mean) / (self.running_var + self.eps)**(0.5)
            ret = self.theta_mu + self.theta_sigma * normalized_x
            return ret
    
