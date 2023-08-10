import util, dataset, model
import torch
import numpy as np

d = dataset.ECGDataset('../data/simg/train.h5')
i, t = d[0]
m = model.CNN()
o = m(i)
