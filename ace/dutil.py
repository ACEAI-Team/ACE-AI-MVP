import numpy as np
import torch
import matplotlib.pyplot as plt


def import_data(path):
  data = np.loadtxt(path, delimiter=',')
  inputs = data[:, :-1]
  outputs = data[:, -1]
  return inputs, outputs

def line(image, point1, point2):
  width = point2[0] - point1[0] + 1
  height = point2[1] - point1[1]
  steps = np.arange(width + 1) * (height // width) + point1[1]
  steps[-1] += height % width
  print(width, steps)
  sign = np.sign(height)
  sign = sign if sign else 1
  last = np.full((512, 256), -1)
  for i in range(width):
    start = steps[i]
    end = steps[i + 1]
    print('draw from', start, 'to', end + sign)
    image[point1[0] + i, start:end + sign:sign] = 1
    if (last == image).all():
      print('NOTHING DRAWN')
    last = np.copy(image)
  return image

def plotter(points, res=(512, 256)):
  image = np.zeros(res)
  for point1, point2 in zip(points, points[1:]):
    print(point1, 'to', point2)
    image = line(image, point1, point2)
    #plt.imshow(image.T, origin='lower')
    #plt.show()
  return image

def grapher(values, res=(512, 256)):
  width = values.shape[0]
  x = (np.arange(width) / (width - 1) * (res[0] - 1)).astype(np.int32)
  y = (values * (res[1] - 1)).astype(np.int32)
  points = np.column_stack((x, y))
  return points


inputs, outputs = import_data('../data/mitbih_test.csv')
print('loaded data')
points = grapher(inputs[0])
image = plotter(points)
plt.imshow(image.T, origin='lower')
plt.show()
