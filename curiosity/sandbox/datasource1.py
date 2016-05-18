from StringIO import StringIO
from PIL import Image
import numpy as np
import h5py
import json
import zmq

def handle_message(sock):
    info = sock.recv()
    nstr = sock.recv()
    narray = np.asarray(Image.open(StringIO(nstr)).convert('RGB'))
    ostr = sock.recv()
    oarray = np.asarray(Image.open(StringIO(ostr)).convert('RGB'))
    imstr = sock.recv()
    imarray = np.asarray(Image.open(StringIO(imstr)).convert('RGB'))
    return [info, narray, oarray, imarray]


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)

print("connecting...")
sock.connect("tcp://18.93.15.188:23042")
print("...connected")
sock.send(json.dumps({'n': 4, 'msg': {"msg_type": "CLIENT_JOIN"}}))
print("...joined")

print('creating sock2 ...')
sock2 = ctx.socket(zmq.REP)
sock2.bind('tcp://18.93.3.135:23042')
print('... bound')

N = 1024000

path = '/data2/datasource1'
file = h5py.File(path, mode='a')
valid = file.require_dataset('valid', shape=(N,), dtype=np.bool)
images = file.require_dataset('images', shape=(N, 256, 256, 3), dtype=np.uint8)
normals = file.require_dataset('normals', shape=(N, 256, 256, 3), dtype=np.uint8)
objects = file.require_dataset('objects', shape=(N, 256, 256, 3), dtype=np.uint8)

rng = np.random.RandomState(0)

while True:
    msg = sock2.recv_json()
    print(msg)
    bn = msg['batch_num']
    bsize = msg['batch_size']
    start = (bn * bsize) % N
    end = ((bn + 1) * bsize) % N

    if not valid[start: end].all():
        print("Getting batch %d new" % bn)
        ims = []
        objs = []
        norms = []
        for i in range(bsize):
            info, narray, oarray, imarray = handle_message(sock)
            msg = {'n': 4,
                   'msg': {"msg_type": "CLIENT_INPUT",
                           "get_obj_data": False,
                           "actions": []}}
            if i % 5 == 0:
                print('teleporting at %d ... ' % i)
                msg['msg']['teleport_random'] = True
                a, b, c = [.3 * rng.uniform(), 0.15 * rng.uniform(), 0.3 * rng.uniform()]
                d, e, f = [0, 2 * rng.binomial(1, .5) - 1, 0]
            else:
                msg['msg']['vel'] = [a, b, c]
                msg['msg']['ang_vel'] = [d, e, f]
            ims.append(imarray)
            norms.append(narray)
            objs.append(oarray)
            sock.send_json(msg) 
        ims = np.array(ims)
        norms = np.array(norms)
        objs = np.array(objs)
        images[start: end] = ims
        normals[start: end] = norms
        objects[start: end] = objs
        valid[start: end] = True
    print("Sending batch %d" % bn)
    ims = images[start: end] 
    norms = normals[start: end] 
    objs = objects[start: end]
    send_array(sock2, ims, flags=zmq.SNDMORE)
    send_array(sock2, norms, flags=zmq.SNDMORE)
    send_array(sock2, objs)
    

