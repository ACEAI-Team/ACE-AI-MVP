import numpy as np
import matplotlib.pyplot as plt
import h5py


def import_data(path):
  data = np.loadtxt(path, delimiter=',')
  inputs = data[:, :-1]
  outputs = data[:, -1]
  return inputs, outputs

def plot_sect(res, vals, max_val=1):
  max_x, max_y = res
  val_len = vals.shape[0]
  x = np.arange(val_len) / (val_len - 1) * (max_x - 1)
  y = vals / max_val * (max_y - 1)
  points = np.stack((x, y), axis=-1).astype(np.int64)
  sects = np.stack((points[:-1], points[1:]), axis=-2)
  return sects

def draw_line(max_range, points):
  point_dif = np.diff(points, axis=0)[0]
  swap_axes = abs(point_dif[1]) > point_dif[0]
  if swap_axes:
    points = np.flip(points, axis=-1)
    point_dif = np.flip(point_dif)
  slope = point_dif[1] / point_dif[0]
  y_int = points[0][1] - slope * points[0][0]
  sign = -1 if point_dif[0] < 0 else 1
  x = max_range[points[0][0]: points[1][0]: sign]
  y = (x * slope + y_int).astype(np.int64)
  if swap_axes:
    return y, x
  return x, y
v_draw_line = np.vectorize(draw_line, signature='(a),(b,c)->(),()', otypes=[np.ndarray, np.ndarray])

def graph(res, vals, max_val=1):
  max_range = np.arange(res[0] if res[0] > res[1] else res[1], dtype=np.int64)
  points = plot_sect(res, vals, max_val)
  x, y = v_draw_line(max_range, points)
  x = np.hstack(x)
  y = np.hstack(y)
  canvas = np.zeros(res)
  canvas[x, y] = 1
  canvas[*points[-1, -1]] = 1
  return canvas
v_graph = np.vectorize(graph, signature='(2),(a),()->(b,c)')

def show_img(image):
  plt.imshow(image.T, origin='lower')
  plt.show()


if __name__ == '__main__':
  in_dir = input('Input Dir (to csv file): ')
  out_dir = input('Output Dir (output hdf5 file no need for extension): ')
  res = eval(input('Image resolution: '))
  print('Importing data...')
  inputs, outputs = import_data(in_dir)
  graph_amount = len(outputs)
  batch = int(input(f'Imported {graph_amount} graphs\nBatch Amount (how many to convert at a time): '))
  with h5py.File(out_dir + '.h5', 'a') as f:
    dset_out = f.create_dataset('outputs', data=outputs, chunks=True)
    dset_in = f.create_dataset('inputs', shape=(graph_amount,) + res, chunks=True)
    for b in range(0, graph_amount, batch):
      print(f'Converting batch {b}/{graph_amount}...')
      images = v_graph(res, inputs[b: b + batch], 1)
      dset_in[b: b + batch] = images
      print(f'Saved batch {b}.')
  print('FINISHED!')
