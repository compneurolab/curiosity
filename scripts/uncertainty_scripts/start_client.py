'''
Just to quickly run some tests, client side.

'''

import socket
import zmq
import json

print('enter addy')
addy = raw_input()


ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('tcp://' + addy + ':' + str(23402))


msg0 = json.dumps({'test' : 0})
sock.send_json(msg0)