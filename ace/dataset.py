import h5py
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
  def __init__(self, dset_dir):
    self.dset_dir = dset_dir

  def __len__(self):
    with h5py.File(self.dset_dir, 'r') as f:
      return len(f['outputs'])

  def __getitem__(self, index):
    with h5py.File(self.dset_dir, 'r') as f:
      inputs =  torch.tensor(f['inputs'][index])
      outputs =  torch.tensor(f['outputs'][index])
