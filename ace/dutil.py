import numpy as np
import torch
import matplotlib.pyplot as plt


def import_data(path):
  data = np.loadtxt(path, delimiter=',')
  inputs = data[:, :-1]
  outputs = data[:, -1]
  return inputs, outputs

def line(image, x1, y1, x2, y2):
  width = x2 - x1 + 1
  height = y2 - y1
  steps = np.arange(width + 1) * (height // width) + y1
  steps[-1] += height % width
  for x, y in zip(range(x1, x2 + 1), steps):
    return steps

def plotter(points, res):
  image = np.zeros((res, res))

def grapher(values, res=(512, 256)):
  width = values.shape[0]
  x = (np.arange(width) / (width - 1) * (res[0] - 1)).astype(np.int32)
  y = (values * (res[1] - 1)).astype(np.int32)
  return x, y


inputs, outputs = import_data('../data/mitbih_test.csv')
x, y = grapher(inputs[0])
i = np.zeros((512, 256))
steps = line(i, x[0], y[0], x[1], y[1])
