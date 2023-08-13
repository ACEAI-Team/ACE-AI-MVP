import h5py
import torch
from torch.utils.data import Dataset

class OldDataset(Dataset):
  def __init__(self, dset_dir):
    self.dset_dir = dset_dir
    self.file = h5py.File(self.dset_dir, 'r')

  def __len__(self):
    return len(self.file['outputs'])

  def __getitem__(self, index):
    inputs = torch.tensor(self.file['inputs'][index])
    outputs = torch.tensor(self.file['outputs'][index])
    return inputs.unsqueeze(0), outputs

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.file.close()


class ECGDataset(Dataset):
  def __init__(self, dset_dir):
    self.dset_dir = dset_dir
    with h5py.File(self.dset_dir, 'r') as file:
      self.inputs = file['inputs'][:]
      self.outputs = file['outputs'][:]

  def __len__(self):
    return len(self.outputs)

  def __getitem__(self, index):
    inputs = torch.tensor(self.inputs[index]).unsqueeze(0)
    outputs = torch.tensor(self.outputs[index])
    return inputs, outputs
