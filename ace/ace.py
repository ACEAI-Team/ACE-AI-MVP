import util, model
import bluetooth
import torch
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def getval(sock):
    data = b''
    while data != b'\n':
        data = sock.recv(1)
    return float(sock.recv(5))

def getdata():
    while 1:
        val = getval(sock)
        stream.pop(0)
        stream.append(val)

def vistream(stream):
    message = '|\t'
    for y in range(20):
        for x in range(10):
            message += f'{stream[y * 10 + x]}\t'
        message += '|\n|\t'
    return message

def predictor():
    categories = ['Non-ecotopiic beats', 'Supraventricular ectopic beats', 'Ventricular ectopic beats', 'Fusion Beats', 'Unknown Beats']
    while 1:
        inputs = torch.tensor(stream[-320:])
        inputs = torch.nn.functional.avg_pool1d(inputs.unsqueeze(0), kernel_size=12, stride=1)[0]
        inputs -= inputs.min()
        in_max = inputs.max().float()
        inputs /= in_max if in_max else 1.
        image = torch.tensor(util.graph((256, 256), inputs)).float()
        with torch.no_grad():
          logits = cnn(image.unsqueeze(0).unsqueeze(0))
        probabilities = torch.nn.functional.softmax(logits)[0].numpy()
        message = ''
        for category, probability in zip(categories, probabilities):
            message += f'\n{category}: {int(probability * 10000)/100}%'
        message += f'\n\n\nChoice: {categories[np.argmax(probabilities)]}'
        print(message)

def update(i):
    ax.clear()
    ax.plot(stream)


mac_address = "B8:D6:1A:6B:32:7A"
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((mac_address, 1))

stream = [0] * 1000
model_file = input('Model file: ')
model_file = model_file if model_file else 'ace.ckpt'
cnn = model.CNN()
cnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
cnn.eval()

fig, ax = plt.subplots()
plt.ion() # turn on interactive mode
ani = FuncAnimation(fig, update, interval=10)

t1 = threading.Thread(target=getdata)
t2 = threading.Thread(target=predictor)
t1.setDaemon(True)
t2.setDaemon(True)
t1.start()
t2.start()

plt.show(block=True)

t1.join()
t2.join()
