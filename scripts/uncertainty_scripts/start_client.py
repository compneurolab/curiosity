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
msg0_back = sock.recv_json()



msg1 = '0'
sock.send(msg1)


msg0_back = json.loads(msg0_back)
print(msg0_back)

