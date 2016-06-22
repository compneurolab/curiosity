"""

"""
import os
from StringIO import StringIO
from PIL import Image
import numpy as np
import h5py
import json
import zmq

from curiosity.utils.io import (handle_sock,
                                send_array, 
                                recv_array)


ctx = zmq.Context()
sock2 = ctx.socket(zmq.REP)
sock2.bind('tcp://18.93.3.135:23044')

file = None

def initialize(path)
    global file
    if file is None:
        file = h5py.File(path, mode='r')
    else:
        assert path == file.filename, (path, file.filename)


while True:
    msg = sock2.recv_json()
    print(msg)
    initialize(msg['path'])
    keys = msg['keys']
    if 'batch_size' in msg:
        N = file[keys[0]].shape[0]
        bn = msg['batch_num']
        batch_size = msg['batch_size']
        start = (bn * bsize) % N
        end = ((bn + 1) * bsize - 1) % N + 1
        sl = slice(start, end)
    else:
        sl = slice(None)
    print("Sending batch %d" % bn)
    for ind, k in enumerate(keys):
        data = file[k][sl]
        if ind < len(keys) - 1:
            send_array(sock2, data, flags=zmq.SNDMORE)
        else:
            send_array(sock2, data)
