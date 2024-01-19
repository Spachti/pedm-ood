import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from pedm.nn_models.mdrnn import MDRNN

# This constant is important for the expand/flatten function
# It defines how many steps per episode the agent takes
STEPS = 1000

class MDRNNEnsemble(MDRNN):
    '''
    This is just a simple class that enables ensembling for the MDRNN model.
    It also implements the neccessary expand and flatten functions for the
    detector testing using particles.
    '''
    def __init__(self, ens_size=1):
        super().__init__()
        self.ens_size = ens_size
        self.MDRNNs = [MDRNN() for _ in range(self.ens_size)]

    def fit(self, X_train, y_train, X_val, y_val, *args, **kwargs):
        return [mdrnn.fit(X_train, y_train, X_val, y_val, *args, **kwargs)
                for mdrnn in self.MDRNNs]
    
    def forward(self, states, actions):
        return [mdrnn.forward(sts, acts) for mdrnn, sts, acts in
                zip(self.MDRNNs, states, actions)]
    
    def sample(self, mean, var, pi):
        # mus: shape(batch_size, seq_len, gaussians, obs_dim)
        # sigmas: shape(batch_size, seq_len, gaussians, obs_dim)
        # pi: shape(batch_size, seq_len, self.gaussians)

        pi = pi.unsqueeze(-1)
        mean = (mean * pi).sum(dim=-2).squeeze()
        var = (var * pi).sum(dim=-2).squeeze()

        # BS, SL, _, OS = mean.shape
        # pis = Categorical(pi).sample().view(BS, SL, 1, 1).expand(-1, -1, -1, OS)
        # mean = mean.detach().gather(-2, pis).squeeze()
        # var = var.detach().gather(-2, pis).squeeze()

        normal_dist = Normal(mean, var)
        samples = normal_dist.sample()
# 
        # samples = mean + torch.randn_like(mean, device=self.device) * var.sqrt()
        
        return samples
    
    def _expand(self, arr):
        dim = arr.shape[-1]
        reshaped = arr.view(STEPS, -1, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.view(self.ens_size, -1, STEPS, dim)
        return reshaped

    def _flatten(self, arr):
        dim = arr.shape[-1]
        reshaped = arr.reshape(-1, STEPS, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.reshape(-1, dim)
        return reshaped
    
    def _unflatten(self, arr, n_part):
        dim = arr.shape[-1]
        n_opt = arr.shape[0] // n_part
        reshaped = arr.reshape(n_opt, self.ens_size, -1, dim)
        return reshaped