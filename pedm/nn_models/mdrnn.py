import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from utils.callbacks import BaseCallback
from utils.gpu_utils import get_device
from utils.loader import RolloutDataset
import numpy as np
import csv

# here I defined some constants to make the code work
# batch size, observation size, action size, hidden layer size (LSTM), sequence length
# has to be adapted to each environment, not the prettiest solution but it works...
BSIZE, OSIZE, ASIZE, HSIZE, SEQ_LEN = 8, 20, 7, 256, 150


# This is the the loss function implemented in the paper "World Models"
# After extensive testing I found that the PEDM loss works better for the OODD
def gmm_loss(batch, mean, var, pi, reduce=True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mean, var)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = torch.log(pi) + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdims=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class MDRNN(nn.Module):
    def __init__(self, normalize_data=True, lr=1e-3, device="auto"):
        super().__init__()

        self.obs = OSIZE
        self.actions = ASIZE
        self.hiddens = HSIZE
        self.gaussians = 5  # Number of Gaussian dists. that are used to model the dist. of s_t+1

        ### Here I tried embedding the input, did not work (as we discussed) ###
        # self.embedding = nn.Sequential(
        #     nn.Linear(self.obs + self.actions, self.embed),
        #     nn.SiLU(),
        #     nn.Linear(self.embed, self.embed),
        #     nn.SiLU()
        #     )
        
        self.rnn = nn.LSTM(self.obs + self.actions, self.hiddens, batch_first=True)     # Recurrent Neural Network
        self.gmm_linear = nn.Linear(self.hiddens, (2 * self.obs + 1) * self.gaussians)  # Mixed Density Network or Gaussian Mixture Model

        self.inputs_mu = nn.Parameter(torch.zeros(1, self.obs + self.actions),
                                      requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, self.obs + self.actions),
                                         requires_grad=False)
        
        self.max_logvar = nn.Parameter(torch.ones(1, self.gaussians, self.obs, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.gaussians, self.obs, dtype=torch.float32) * 10.0)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = get_device(device)
        self.to(self.device)
        # print(sum(p.numel() for p in self.parameters() if p.requires_grad))
        self.normalize_data = normalize_data
        self.domain_rand = False

    def norm_data(self, X_train):
        mu = torch.mean(X_train, dim=0, keepdims=True)
        sigma = torch.std(X_train, dim=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.inputs_mu.data = mu.to(self.device).float()
        self.inputs_sigma.data = sigma.to(self.device).float()

    def forward(self, obs, actions, ret_logvar=False):
        batch_size, seq_len, _ = obs.shape

        # Here I tried implementing domain randomization
        # Turns out I only added Gaussian noise to some randomly chosen states
        if self.domain_rand:
            mask = torch.rand(batch_size, seq_len, 1)
            mask = torch.round(mask).to(self.device)
            domain_rand = torch.randn_like(obs)
            domain_rand = domain_rand * mask
            with torch.no_grad():
                obs += domain_rand

        inputs = torch.cat([obs, actions], dim=-1)

        if self.normalize_data:
            inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        rnn_outputs, _ = self.rnn(inputs)
        gmm_outputs = self.gmm_linear(rnn_outputs)

        stride = self.gaussians * self.obs

        mean = gmm_outputs[:, :, :stride]
        mean = mean.view(batch_size, seq_len, self.gaussians, self.obs)

        logvar = gmm_outputs[:, :, stride:2 * stride]
        logvar = logvar.view(batch_size, seq_len, self.gaussians, self.obs)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        # This is the main difference to the PEDM's distribution estimation
        # For each of the Gaussian dists. (self.gaussians) it also predicts a pi-value
        # that acts as a weighting of each individual Gaussian dist. 
        # The pi-values have to sum to 1 --> softmax
        pi = gmm_outputs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(batch_size, seq_len, self.gaussians)
        pi = F.softmax(pi, dim=-1)

        if ret_logvar:
            return mean, logvar, pi
        else:
            return mean, torch.exp(logvar), pi
    
    def fit(self, X_train, y_train, X_val, y_val, n_train_epochs, callback=BaseCallback()):
        callback.init_callback(_locals=locals())
        callback.on_train_begin(_locals=locals())

        X_train = torch.from_numpy(X_train).to(self.device).float()
        y_train = torch.from_numpy(y_train).to(self.device).float()
        X_val = torch.from_numpy(X_val).to(self.device).float()
        y_val = torch.from_numpy(y_val).to(self.device).float()

        # I implemented a custom PyTorch Dataset to make handling the sequential data possible (loader.py in utils)
        # This also sped up training by quite a margin!
        train_data = RolloutDataset(X_train, y_train, SEQ_LEN, OSIZE, ASIZE)
        val_data = RolloutDataset(X_val, y_val, SEQ_LEN, OSIZE, ASIZE)
        train_loader = DataLoader(train_data, batch_size=BSIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BSIZE)

        if self.normalize_data:
            self.norm_data(X_train)

        train_loss = []
        val_loss = []
        for ep in range(n_train_epochs):
            callback.on_ep_begin(_locals=locals())
            ep_train_loss = self.data_pass(train=True, loader=train_loader)
            ep_val_loss = self.data_pass(train=False, loader=val_loader)
            train_loss.append(ep_train_loss)
            val_loss.append(ep_val_loss)
            if callback.on_ep_end(_locals=locals()):
                break

        # With this I evaluated loss curves, I know that there are better tools for this
        # but I don't feel so comfortable with them yet
        with open('losses.csv', 'w', newline='') as file:
            writer = csv.writer(file)    
            writer.writerow(train_loss)
            writer.writerow(val_loss)

        result = callback.on_train_end(_locals=locals())
        # self._val_threshold(X_val=X_val, y_val=y_val)
        return result if result else ep_train_loss, ep_val_loss
    
    def get_loss(self, obs, action, next_obs):
        '''
        This is the loss function of the OODD paper adapted to a MDN.
        I weigh the mean and logvar here with their respective pi-values
        and then I add them together to get the final dist. that is used
        to calculate the loss.
        '''
        mean, logvar, pi = self.forward(obs, action, ret_logvar=True)
        # mean, logvar = self.sample_from_pi(mean, logvar, pi)
        pi = pi.unsqueeze(-1)
        mean = (mean * pi).sum(dim=-2)
        logvar = (logvar * pi).sum(dim=-2)
        inv_var = torch.exp(-logvar)
        loss = ((mean - next_obs) ** 2) * inv_var + logvar
        loss = loss.mean(-1).mean(-1).sum()

        return loss
    
    def get_paper_loss(self, obs, action, next_obs):
        '''
        This was the loss for the paper but as mentioned in the
        beginnig, this did not really work well.
        '''
        mean, var, pi = self.forward(obs, action)
        gmm = gmm_loss(next_obs, mean, var, pi)
        loss = gmm / OSIZE
        return loss
    
    def sample_from_pi(self, mean, logvar, pi):
        '''
        Here I first sampled from pi to get the mean and the log-variance.
        This does not really make sense because we want the superposition of
        the Gaussian dists. instead of only one in order to be able to describe 
        the underlying dist. of s_t+1 as well as possible.
        '''
        BS, SL, _, OS = mean.shape
        pis = Categorical(pi).sample().view(BS, SL, 1, 1).expand(-1, -1, -1, OS)
        mean = mean.gather(-2, pis).squeeze()
        logvar = logvar.gather(-2, pis).squeeze()
        return mean, logvar
    
    def data_pass(self, train, loader):
        '''
        This is the training function. Nothing special going on here.
        '''
        if train:
            self.train()
        else:
            self.eval()

        loader = loader

        acc_loss = []
        # cum_loss = 0

        for _, data in enumerate(loader):
            loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
            # loss = 0
            obs, action, next_obs = data
            if train:
                train_loss = self.get_loss(obs, action, next_obs)
                # train_loss = self.get_paper_loss(obs, action, next_obs)
                loss += train_loss
                acc_loss.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            else:
                with torch.no_grad():
                    mean, _, pi = self.forward(obs, action)
                    pi = pi.unsqueeze(-1)
                    mean = (mean * pi).sum(dim=-2)
                    se_loss = (mean - next_obs) ** 2
                    mse_loss = se_loss.mean(-1).mean(-1).sum()
    
        return np.mean(acc_loss) if train else mse_loss.cpu().numpy()
        