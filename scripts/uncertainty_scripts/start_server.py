'''
Just to quickly run some tests, server side.

'''

import socket
import zmq
import json

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('google.com', 80))
addy = s.getsockname()[0]
s.close()

print(addy)

ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind('tcp://' + addy + ':' + str(23402))

msg0 = sock.recv_json()
print(msg0)

