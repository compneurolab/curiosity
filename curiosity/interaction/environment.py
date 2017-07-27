'''
Environment object for interaction with 3World
'''


import socket
import numpy as np
import zmq
import copy
import pymongo
from bson.objectid import ObjectId
from PIL import Image
from scipy.misc import imresize
from curiosity.interaction.tdw_client import TDW_Client
SELECTED_BUILD = ':\Users\mrowca\Desktop\world\one_world.exe'

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import json
import cv2
import copy
import cPickle


synset_for_table = [[u'n04379243']]
rollie_synsets = [[u'n03991062'], [u'n02880940'], [u'n02946921'], [u'n02876657'], [u'n03593526']]
other_vaguely_stackable_synsets = [[u'n03207941'], [u'n04004475'], [u'n02958343'], [u'n03001627'], [u'n04256520'], [u'n04330267'], [u'n03593526'], [u'n03761084'], [u'n02933112'], [u'n03001627'], [u'n04468005'], [u'n03691459'], [u'n02946921'], [u'n03337140'], [u'n02924116'], [u'n02801938'], [u'n02828884'], [u'n03001627'], [u'n04554684'], [u'n02808440'], [u'n04460130'], [u'n02843684'], [u'n03928116']]

shapenet_inquery = {'type': 'shapenetremat', 'has_texture': True, 'version': 0, 'complexity': {'$exists': True}, 'center_pos': {'$exists': True}, 'boundb_pos': {'$exists': True}, 'isLight': {'$exists': True}, 'anchor_type': {'$exists': True}, 'aws_address': {'$exists': True}}
dosch_inquery = {'type': 'dosch', 'has_texture': True, 'version': 1, 'complexity': {'$exists': True}, 'center_pos': {'$exists': True}, 'boundb_pos': {'$exists': True}, 'isLight': {'$exists': True}, 'anchor_type': {'$exists': True}, 'aws_address': {'$exists': True}}


# default_inquery     = {'type': 'shapenetremat', 'has_texture': True, 'complexity': {'$exists': True}, 'center_pos': {'$exists': True}, 'boundb_pos': {'$exists': True}, 'isLight': {'$exists': True}, 'anchor_type': {'$exists': True}, 'aws_address': {'$exists': True}}


default_keys = ['boundb_pos', 'isLight', 'anchor_type', 'aws_address', 'complexity', 'center_pos']


table_query = copy.deepcopy(shapenet_inquery)
table_query['synset'] = {'$in' : synset_for_table}
rolly_query = copy.deepcopy(shapenet_inquery)
rolly_query['synset'] = {'$in' : rollie_synsets}
other_reasonables_query = copy.deepcopy(shapenet_inquery)
other_reasonables_query['synset'] = {'$in' : other_vaguely_stackable_synsets}

query_dict = {'SHAPENET' : shapenet_inquery, 'ROLLY' : rolly_query, 'TABLE' : table_query, 'OTHER_STACKABLE' : other_reasonables_query}


example_scene_info = [
        {
        'type' : 'SHAPENET',
        'scale' : .4,
        'mass' : 1.,
        'scale_var' : .01,
        'num_items' : 1,
        }
        ]


def query_results_to_unity_data(query_results, scale, mass, var = .01, seed = 0):
	item_list = []
	for i in range(len(query_results)):
		res = query_results[i]
		item = {}
		item['type'] = res['type']
		item['has_texture'] = res['has_texture']
		item['center_pos'] = res['center_pos']
		item['boundb_pos'] = res['boundb_pos']
		item['isLight'] = res['isLight']
		item['anchor_type'] = res['anchor_type']
		item['aws_address'] = res['aws_address']
		item['mass'] = mass
		item['scale'] = {"option": "Absol_size", "scale": scale, "var": var, "seed": seed, 'apply_to_inst' : True}
		item['_id_str'] = str(res['_id'])
		item_list.append(item)
	return item_list



def init_msg(n_frames):
        msg = {'n': n_frames, 'msg': {"msg_type": "CLIENT_INPUT", "get_obj_data": True, "send_scene_info" : True, "actions": []}}
        msg['msg']['vel'] = [0, 0, 0]
        msg['msg']['ang_vel'] = [0, 0, 0]
        return msg



def test_action_to_message_fn(action, env):
	msg = init_msg(env.num_frames_per_msg)
	if action == 0:
		msg['msg']['vel'] = [0, 0, .5]
		msg['msg']['action_type'] = 'FORWARD'
	elif action == 1:
		msg['msg']['ang_vel'] = [0, .5, 0]
		msg['msg']['action_type'] = 'ROT'
	elif action == 2:
		msg['msg']['ang_vel'] = [0, -.5, 0]
		msg['msg']['action_type'] = 'COUNTER_ROT'
	else:
		raise Exception('Action not implemented!')
	return msg


def normalized_action_to_ego_force_torque(action, env, limits, wall_safety = None):
	'''
		Sends message given forward vel, y-angular speed, force, torque.
		If wall_safety is a number, stops the agent from stepping closer to the wall from wall_safety.
	'''
	msg = init_msg(env.num_frames_per_msg)
	limits = np.array(limits)
	action = np.copy(action)
	action_normalized = action
	action = action * limits
	agent_vel = action[0]
	if wall_safety is not None:
		#check for wall safety
		proposed_next_position = np.array(env.observation['info']['avatar_position']) + np.array(env.observation['info']['avatar_forward']) * agent_vel
		if (proposed_next_position[0]  < wall_safety or proposed_next_position[0] > env.ROOM_WIDTH - .5 - wall_safety
					or proposed_next_position[2] < wall_safety or proposed_next_position[2] > env.ROOM_LENGTH - .5 - wall_safety):
			print('wall safety!')
			agent_vel = 0.
			action_normalized[0] = 0.
	msg['msg']['vel'] = [0, 0, agent_vel]
	msg['msg']['ang_vel'] = [0, action[1], 0]

	available_objects = [o for o in env.observation['info']['observed_objects'] if not o[5] and int(o[1]) != -1 and not o[4]]
	if len(available_objects) > 1:
		raise Exception('This action parametrization only meant for one object')
	elif len(available_objects) == 1:
		obj_id = available_objects[0][1]
		oarray = env.observation['objects1']
		if type(oarray) == list:
			oarray = oarray[-1]
		#see if the object is in view
		oarray1 = 256**2 * oarray[:, :, 0] + 256 * oarray[:, :, 1] + oarray[:, :, 2]
		xs, ys = (oarray1 == obj_id).nonzero()
		#if not in view
		if len(xs) == 0:
			msg['msg']['action_type'] = 'NO_OBJ_ACT'
			for i in range(2, 8):
				action_normalized[i] = 0.
		#if in view
		else:
			#set action_pos equal to the seen center of mass. this doesn't matter
			seen_cm = np.round(np.array(zip(xs, ys)).mean(0))
			msg['msg']['action_type'] = 'OBJ_ACT'
			msg_action = {}
			msg_action['use_absolute_coordinates'] = True
			msg_action['force'] = list(action[2:5])
			msg_action['torque'] = list(action[5:])
			msg_action['id'] = str(obj_id)
			msg_action['object'] = str(obj_id)
			msg_action['action_pos'] = list(map(float, seen_cm))
			msg['msg']['actions'].append(msg_action)
	else:
		print('Warning! object not found!')
		for i in range(2, 8):
			msg['msg']['action_type'] = 'OBJ_NOT_PRESENT'
			action_normalized[i] = 0.
	return msg, action_normalized


def handle_message_new(sock, msg_names, write = False, outdir = '', imtype =  'png', prefix = ''):
    info = sock.recv()
    data = {'info': info}
    # Iterate over all cameras
    for cam in range(len(msg_names)):
        for n in range(len(msg_names[cam])):
            # Handle set of images per camera
            imgstr = sock.recv()
            imgarray = np.asarray(Image.open(StringIO(imgstr)).convert('RGB'))
            field_name = msg_names[cam][n] + str(cam+1)
            assert field_name not in data, \
                    ('duplicate message name %s' % field_name)
            data[field_name] = imgarray
    return data



def handle_message(sock, write=False, outdir='', imtype='png', prefix=''):
    # Handle info
    print('recv1')
    info = sock.recv()
    # print("got message")
    # Handle first set of images from camera 1
    print('recv2')
    nstr = sock.recv()
    narray2 = np.asarray(Image.open(StringIO(nstr)).convert('RGB'))
    print('recv3')
    ostr = sock.recv()
    oarray2 = np.asarray(Image.open(StringIO(ostr)).convert('RGB'))
    print('recv4')
    dstr = sock.recv()
    darray2 = np.asarray(Image.open(StringIO(dstr)).convert('RGB'))
    print('recv5')
    imstr = sock.recv()
    imarray2 = np.asarray(Image.open(StringIO(imstr)).convert('RGB'))
    # Handle second set of images from camera 2
    print('recv6')
    nstr = sock.recv()
    narray = np.asarray(Image.open(StringIO(nstr)).convert('RGB'))
    print('recv7')
    ostr = sock.recv()
    oarray = np.asarray(Image.open(StringIO(ostr)).convert('RGB'))
    print('recv8')
    dstr = sock.recv()
    darray = np.asarray(Image.open(StringIO(dstr)).convert('RGB'))
    print('recv9')
    imstr = sock.recv()
    imarray = np.asarray(Image.open(StringIO(imstr)).convert('RGB'))

    im = Image.fromarray(imarray)
    im2 = Image.fromarray(imarray2)
    imo = Image.fromarray(oarray)

    #dim = Image.fromarray(darray2)
    #print(outdir, prefix, imtype)
    #dim.save(os.path.join('/Users/damian/Desktop/test_images/new/', 'depth_%s.%s' % (prefix, imtype)))

    if write:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        im.save(os.path.join(outdir, 'image_%s.%s' % (prefix, imtype)))
        im2.save(os.path.join(outdir, '2image_%s.%s' % (prefix, imtype)))
        imo.save(os.path.join(outdir, 'objects_%s.%s' % (prefix, imtype)))
        #with open(os.path.join(outdir, 'image_%s.%s' % (prefix, imtype)), 'w') as _f:
        #    _f.write(imstr)
        #with open(os.path.join(outdir, 'objects_%s.%s' % (prefix, imtype)), 'w') as _f:
        #    _f.write(ostr)
        #with open(os.path.join(outdir, 'normals_%s.%s' % (prefix, imtype)), 'w') as _f:
        #    _f.write(nstr)
        if '_0' in prefix:
            with open(os.path.join(outdir, 'info_%s.json' % prefix), 'w') as _f:
                _f.write(info)
    return [info, narray, oarray, darray, imarray, narray2, oarray2, darray2, imarray2]


SHADERS = [{'DisplayDepth': 'png'}, {'GetIdentity' : 'png'}, {'Images' : 'png'}]
HDF5_NAMES = [{'DisplayDepth' : 'depths'}, {'GetIdentity' : 'objects'}, {'Images' : 'images'}]

SHADERS_DEPTH = [{'DisplayDepth': 'png'}, {'GetIdentity' : 'png'}]
HDF5_NAMES_DEPTH = [{'DisplayDepth' : 'depths'}, {'GetIdentity' : 'objects'}]

SHADERS_LONG = [{"DisplayNormals": "png"}, {"GetIdentity": "png"}, {"DisplayDepth": "png"}, {"DisplayVelocity": "png"}, {"DisplayAcceleration": "png"}, {"DisplayJerk": "png"}, {"DisplayVelocityCurrent": "png"}, {"DisplayAccelerationCurrent": "png"}, {"DisplayJerkCurrent": "png"}, {"Images": "jpg"}]
HDF5_NAMES_LONG = [{"DisplayNormals": "normals"}, {"GetIdentity": "objects"}, {"DisplayDepth": "depths"}, {"DisplayVelocity": "velocities"}, {"DisplayAcceleration": "accelerations"}, {"DisplayJerk": "jerks"},  {"DisplayVelocityCurrent": "velocities_current"}, {"DisplayAccelerationCurrent": "accelerations_current"}, {"DisplayJerkCurrent": "jerks_current"}, {"Images": "images"}]

#SHADERS = [{"DisplayNormals": "png"}, {"GetIdentity": "png"}, {"DisplayDepth": "png"}, {"DisplayVelocity": "png"}, {"DisplayAcceleration": "png"}, {"DisplayJerk": "png"}, {"Images": "jpg"}]
#HDF5_NAMES = [{"DisplayNormals": "normals"}, {"GetIdentity": "objects"}, {"DisplayDepth": "depths"}, {"DisplayVelocity": "velocities"}, {"DisplayAcceleration": "accelerations"}, {"DisplayJerk": "jerks"}, {"Images": "images"}]

class PeriodicRNGSource:
	'''
	A class for giving rngs to the environment for object selection. Periodically repeats with a fresh one with the same seeds.
	'''
	def __init__(self, period, seed = 0):
		self.period = period
		self.seed = seed
		self.ctr = 0

	def next(self):
		if self.ctr % self.period == 0:
			self.rng = np.random.RandomState(self.seed)
		self.ctr += 1
		return self.rng


class Environment:
	def __init__(self, 
			random_seed,
			 unity_seed, 
			 action_to_message_fn, 
			 USE_TDW = False, 
			 SCREEN_DIMS = (128, 170),
			 room_dims = (20., 20.), #(ROOM_LENGTH, ROOM_HEIGHT)
			state_memory_len = {}, #remembers multiple images and concatenates. ex {'depth' : 2}
			 rescale_dict = {}, #to rescale images after unity. {'depth' : (64, 64)}
			host_address = None,
			msg_names = HDF5_NAMES,
			shaders = SHADERS,
			n_cameras = 2,
			message_memory_len = 2,
			action_memory_len = 2,
			local_pickled_query = None,
			rng_source = None,
			rng_periodicity = None,
			scene_switch_memory_pad = 2, 
			other_data_memory_length = 1
		):
		#TODO: SCREEN_DIMS does nothing right now
		self.rng = np.random.RandomState(random_seed)
		self.SCREEN_HEIGHT, self.SCREEN_WIDTH = SCREEN_DIMS
		self.RANDOM_SEED = unity_seed
		self.shaders = shaders
		self.msg_names = []
		assert len(shaders) == len(msg_names)
		for i in range(len(shaders)):
			assert len(shaders[i].keys()) == 1
			for k in shaders[i]:
				assert k in msg_names[i]
				self.msg_names.append(msg_names[i][k])
		self.msg_names = [self.msg_names] * n_cameras
		self.num_frames_per_msg = 1 + n_cameras * len(shaders)
		self.scene_switch_memory_pad = scene_switch_memory_pad
		#self.num_frames_per_msg = 15
		# #borrowing a hack from old curiosity
		# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		# s.connect(("google.com",80))
		if USE_TDW:
			assert host_address is not None
		if host_address is not None:
			self.host_address = host_address
		else:
			#borrowing a hack from old curiosity
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			s.connect(("google.com",80))
			self.host_address = s.getsockname()[0]
			s.close()
		#setting up socket
		ctx = zmq.Context()
		if USE_TDW:
			self.tc = TDW_Client(self.host_address,
                        initial_command='request_create_environment',
                        description="test script",
                        selected_build=SELECTED_BUILD,  # or skip to select from UI
                        #queue_port_num="23402",
                        get_obj_data=True,
                        send_scene_info=True,
                        num_frames_per_msg=self.num_frames_per_msg,
                        )
		else:
			print ("connecting...")
			self.sock = ctx.socket(zmq.REQ)
			self.sock.connect("tcp://" + self.host_address + ":5556")
			print ("... connected @" + self.host_address + ":" + "5556")
		self.USE_TDW = USE_TDW
		self.not_yet_joined = True
		self.action_to_message_fn = action_to_message_fn
		self.CACHE = {}
		self.local_pickled_query = local_pickled_query
		if local_pickled_query is None:
			self.conn = pymongo.MongoClient(port=22334)
			self.coll = self.conn['synthetic_generative']['3d_models']
		self.COMPLEXITY = 1500#I think this is irrelevant, or it should be. TODO check
		self.NUM_LIGHTS = 4
		self.ROOM_LENGTH, self.ROOM_WIDTH = room_dims
		self.rescale_dict = rescale_dict
		self.state_memory_len = state_memory_len
		self.msg_memory_len = message_memory_len
		self.action_memory_len = action_memory_len
		self.rng_source = rng_source
		if rng_periodicity is not None:
			self.rng_source = PeriodicRNGSource(rng_periodicity, seed = 1)
		self.state_memory = dict((k, [None for _ in range(mem_len)]) for k, mem_len in self.state_memory_len.iteritems())
		self.msg_memory = [None for _ in range(self.msg_memory_len)]
		self.action_memory = [None for _ in range(self.action_memory_len)]
		self.action_post_memory = [None for _ in range(self.action_memory_len)]
		self.other_data_memory = [None for _ in range(other_data_memory_length)]


	def get_items(self, q, num_items, scale, mass, var = .01):
		for _k in default_keys:
			if _k not in q:
				q[_k] = {'$exists': True}
		print('first query')
		if not str(q) in self.CACHE:
			idvals = np.array([str(_x['_id']) for _x in list(self.coll.find(q, projection=['_id']))])
			self.CACHE[str(q)] = idvals
			print('new', q, len(idvals))
		idvals = self.CACHE[str(q)]
		num_ava = len(idvals)
		#might want to just initialize this once
		goodidinds = self.rng.permutation(num_ava)[: num_items] 
		goodidvals = idvals[goodidinds]
		goodidvals = list(map(ObjectId, goodidvals))
		keys = copy.deepcopy(default_keys)
		for _k in q:
			if _k not in keys:
				keys.append(_k)
		print('second query')
		query_res = list(self.coll.find({'_id': {'$in': goodidvals}}, projection=keys))
		print('making items')
		return query_results_to_unity_data(query_res, scale, mass, var = var, seed = self.RANDOM_SEED + 1)

	# update config for next scene switch. like reset() in gym.
	def next_config(self, * round_info):
		#meant so that one can have finer control over 
		if self.rng_source is not None:
			self.rng = self.rng_source.next()
		if self.local_pickled_query is None:
			rounds = [{'items' : self.get_items(query_dict[info['type']], info['num_items'] * 4, info['scale'], info['mass'], info['scale_var']), 'num_items' : info['num_items']} for info in round_info]
		else:
			with open(self.local_pickled_query) as stream:
				rounds = cPickle.load(stream)

		self.config = {
			"environment_scene" : "ProceduralGeneration",
			"random_seed": self.RANDOM_SEED, #Omit and it will just choose one at random. Chosen seeds are output into the log(under warning or log level).
			"should_use_standardized_size": False,
			"standardized_size": [1.0, 1.0, 1.0],
			"complexity": self.COMPLEXITY,
			"random_materials": True,
			"num_ceiling_lights": self.NUM_LIGHTS,
			"intensity_ceiling_lights": 1,
			"use_standard_shader": True,
			"minimum_stacking_base_objects": 5,
			"minimum_objects_to_stack": 5,
			"disable_rand_stacking": 0,
			"room_width": self.ROOM_WIDTH,
			"room_height": 10.0,
			"room_length": self.ROOM_LENGTH,
			"wall_width": 1.0,
			"door_width": 1.5,
			"door_height": 3.0,
			"window_size_width": (5.0/1.618), # standard window ratio is 1:1.618
			"window_size_height": 5.0,
			"window_placement_height": 2.5,
			"window_spacing": 7.0,  #Average spacing between windows on walls
			"wall_trim_height": 0.5,
			"wall_trim_thickness": 0.01,
			"min_hallway_width": 5.0,
			"number_rooms": 1,
			"max_wall_twists": 3,
			"max_placement_attempts": 300,   #Maximum number of failed placements before we consider a room fully filled.
			"grid_size": 0.4,    #Determines how fine tuned a grid the objects are placed on during Proc. Gen. Smaller the number, the
			"use_mongodb_inter": 1, 
			'rounds' : rounds
			}
		if self.not_yet_joined:
			if self.USE_TDW:
				self.tc.load_config(self.config)
				self.tc.load_profile({'screen_width': self.SCREEN_WIDTH, 'screen_height': self.SCREEN_HEIGHT})
				print('about to hit run')
				msg = 'tdw_client_init'
				self.sock = self.tc.run()
				print('run done')
			else:
				print('sending join...')
				msg = {"msg_type" : "CLIENT_JOIN_WITH_CONFIG", "config" : self.config, "get_obj_data" : True, "send_scene_info" : True, 'shaders' : self.shaders}
				self.sock.send_json(msg)
				print('...join sent')
			self.not_yet_joined = False
		else:
			#for i in range(self.num_frames_per_msg):
			#	self.sock.recv()
			print('switching scene...')
			scene_switch_msg = {"msg_type" : "SCENE_SWITCH", "config" : self.config, "get_obj_data" : True, "send_scene_info" : True, 'SHADERS' : self.shaders}
			if self.USE_TDW:
				print('scene switch')
				msg = {"n": self.num_frames_per_msg, "msg": scene_switch_msg}
				self.sock.send_json(msg)
			else:
				msg = scene_switch_msg
				self.sock.send_json(msg)
		observation = self._observe_world()
		print('heard back from unity!')
		self._pad_memory()
		observation, msg, action, action_post, other_dat  = self._memory_postprocess(observation, msg, None, None)
		return observation, msg

	def _pad_memory(self):
		for _ in range(self.scene_switch_memory_pad):
			for k in self.state_memory:
				self.state_memory[k].pop(0)
				self.state_memory[k].append(None)
			self.msg_memory.pop(0)
			self.msg_memory.append(None)
			self.action_memory.pop(0)
			self.action_memory.append(None)
			self.action_post_memory.pop(0)
			self.action_post_memory.append(None)
			self.other_data_memory.pop(0)
			self.other_data_memory.append(None)


	def _memory_postprocess(self, observation, msg, action, action_post, other_data = None):
		for k in self.state_memory:
			self.state_memory[k].pop(0)
			self.state_memory[k].append(observation[k])
		self.msg_memory.pop(0)
		self.msg_memory.append(msg)
		self.action_memory.pop(0)
		self.action_memory.append(action)
		self.action_post_memory.pop(0)
		self.action_post_memory.append(action_post)
		self.other_data_memory.pop(0)
		self.other_data_memory.append(other_data)
		return self.state_memory, self.msg_memory, self.action_memory, self.action_post_memory, self.other_data_memory

	def _observe_world(self):
		self.observation = handle_message_new(self.sock, self.msg_names)
		self.observation['info'] = json.loads(self.observation['info'])
		for (k, shape) in self.rescale_dict.iteritems():
			self.observation[k] = imresize(self.observation[k], shape)
		return self.observation

	def step(self, action, other_data = None):
		#gets message. action_to_message_fn can make adjustments to action
		#other data is included so that we can manage a cache of data all in one place, but the environment otherwise does not interact with it
		msg, action_post = self.action_to_message_fn(action, self)
		if self.USE_TDW:
			self.sock.send_json(msg)
		else:
			self.sock.send_json(msg['msg'])
		observation = self._observe_world()
		return self._memory_postprocess(observation, msg, action, action_post, other_data = other_data)









