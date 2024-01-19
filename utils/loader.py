from torch.utils.data import Dataset

class RolloutDataset(Dataset):
    def __init__(self, X, y, seq_len, osize, asize):
        self.seq_len = seq_len
        self.obs = X[:, :osize].view(-1, self.seq_len, osize)
        self.actions = X[:, osize:].view(-1, self.seq_len, asize)
        self.next_obs = y.view(-1, self.seq_len, osize)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        obs = self.obs[i]
        actions = self.actions[i]
        next_obs = self.next_obs[i]
        
        return obs, actions, next_obs