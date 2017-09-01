'''
Now with new data provider, and 2->2 architecture.
'''
import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import numpy as np

from tfutils import base, optimizer
from curiosity.data.short_long_sequence_data import ShortLongSequenceDataProvider
import curiosity.models.jerk_models as modelsource
import copy
from tqdm import trange
import cPickle

N_STATES = 7
USE_GROUND_TRUTH_FOR_VALIDATION = False
CACHE_NUM = 2
LOCAL = False
if LOCAL:
    DATA_PATH = '/data2/mrowca/datasets/eight_world_dataset/new_tfdata'
    VALDATA_PATH = '/data2/mrowca/datasets/eight_world_dataset/new_tfvaldata'
    CACHE_DIR = '/data2/mrowca/cache' + str(CACHE_NUM)
    STATS_FILE = '/data2/mrowca/datasets/eight_world_dataset/new_stats/stats_std.pkl'
else:
    DATA_PATH = '/mnt/fs1/datasets/eight_world_dataset/new_tfdata'
    VALDATA_PATH = '/mnt/fs1/datasets/eight_world_dataset/new_tfvaldata'
    CACHE_DIR = '/mnt/fs0/mrowca/cache' + str(CACHE_NUM)
    STATS_FILE = '/mnt/fs1/datasets/eight_world_dataset/new_stats/stats_std.pkl'
BIN_PATH = '' #'/mnt/fs1/datasets/eight_world_dataset/'
BIN_FILE = '' #'/mnt/fs1/datasets/eight_world_dataset/bin_data_file.pkl'
SAVE_DIR = '/mnt/fs0/mrowca/results/flex/'

N_GPUS = 1
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 64 #256
TEST_BATCH_SIZE = 1
TIME_SEEN = 1 #2
SHORT_LEN = TIME_SEEN
LONG_LEN = 1 #3
MIN_LEN = 1 #3
NUM_BATCHES_PER_EPOCH = 4000 * 256 / MODEL_BATCH_SIZE
IMG_HEIGHT = 128
IMG_WIDTH = 170
SCALE_DOWN_HEIGHT = 64
SCALE_DOWN_WIDTH = 88
L2_COEF = 200.
EXP_ID = [#'flex2dBott_4', 
'flexBott2LossS7normMask',
#'flex2d_4', 
#'flex_4',
]
#EXP_ID = ['res_jerk_eps', 'map_jerk_eps', 'sym_jerk_eps', 'bypass_jerk_eps']
LRS = [0.001, 0.001, 0.001, 0.001]
n_classes = 3
buckets = 0
min_particle_distance = 0.01
DEPTH_DIM = 32
CFG = [
        #modelsource.particle_2d_bottleneck_cfg(n_classes * DEPTH_DIM, nonlin='relu'),
        #modelsource.particle_bottleneck_cfg(n_classes, nonlin='relu'),
        modelsource.particle_bottleneck_comp_cfg(n_states=N_STATES+2, nonlin='relu'),
        #modelsource.particle_2d_cfg(n_classes * DEPTH_DIM, nonlin='relu'),
        #modelsource.particle_cfg(n_classes, nonlin='relu'),
        ]
CACHE_DIRS = [CACHE_DIR + str(d) for d in range(4)]
SEED = 4

if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

def table_norot_grab_func(path):
    all_filenames = os.listdir(path)
    rng = np.random.RandomState(seed=SEED)
    rng.shuffle(all_filenames)
    print('got to file grabber!')
    return [os.path.join(path, fn) for fn in all_filenames \
            if fn.endswith('.tfrecords')] 
            #and fn.startswith('2:')] \
            #and 'FAST_LIFT' in fn]

def append_it(x, y, step):
    if x is None:
        x = []
    x.append(y)
    return x

def mean_losses_subselect_rest(val_res, skip_num):
    retval = {}
    keys = val_res[0].keys()
    for k in keys:
        if 'loss' in k:
            plucked = [d[k] for d in val_res]
            retval[k] = np.mean(plucked)
        elif 'reference_ids' in k:
            retval[k] = [d[k] for d in val_res]
        else:
            retval[k] = [val_res[i][k] for i in range(len(val_res)) if i % skip_num == 0]
    return retval

def just_keep_everything(val_res):
    keys = val_res[0].keys()
    return dict((k, [d[k] for d in val_res]) for k in keys)

def keep_all(step_results):
    return step_results[0]

SAVE_TO_GFS = []#['pred_vel_flat', 'reference_ids']

def grab_all(inputs, outputs, bin_file = BIN_FILE, 
        num_to_save = 1, gpu_id = 0, **garbage_params):
    retval = {
            'pred_velocity': outputs['pred_velocity'],
            'pred_mask': outputs['pred_mask'],
            'pred_grid': outputs['pred_grid'],
            'next_grid': outputs['next_grid'],
            'next_velocity': outputs['next_velocity'],
            'full_grids': outputs['full_grids'],
            'grid_placeholder': outputs['grid_placeholder'],
            'grids_placeholder': outputs['grids_placeholder'],
            'action_placeholder': outputs['action_placeholder'],
            'actions': outputs['actions'],
            'in_view': outputs['in_view'],
            'is_moving': outputs['is_moving'],
            }
    return retval

save_params = [{
    'host' : 'localhost',
    'port' : 24444,
    'dbname' : 'future_prediction',
    'collname' : 'flex',
    'exp_id' : EXP_ID[0],
    'save_valid_freq' : np.round(256 * 84 * 4 / MODEL_BATCH_SIZE).astype(np.int32),
    'save_filters_freq': np.round(256 * 84 * 4 / MODEL_BATCH_SIZE * 10).astype(np.int32),
    'cache_filters_freq': np.round(256 * 84 * 4 / MODEL_BATCH_SIZE).astype(np.int32),
    'save_metrics_freq': np.round(256 * 84 * 4 / MODEL_BATCH_SIZE / 10).astype(np.int32),
    'save_initial_filters' : False,
    'cache_dir' : CACHE_DIR,
    'save_to_gfs' : SAVE_TO_GFS
}] * N_GPUS

load_params = [{
    'host' : 'localhost',
    'port' : 24444,
    'dbname' : 'future_prediction',
    'collname': 'flex',
    'exp_id' : EXP_ID[0],
    'do_restore': True,
    'load_query': None
}] * N_GPUS

model_params = [{
    'func' : modelsource.flex_comp_model,
    'cfg' : CFG[0],
    'time_seen' : TIME_SEEN,
    'normalization_method' : {
        'actions' : 'minmax'},
    'stats_file' : STATS_FILE,
    'image_height' : IMG_HEIGHT,
    'image_width' : IMG_WIDTH,
    #'num_classes': 60.,
    'gpu_id' : 0,
    'my_test' : True,
    'test_batch_size': TEST_BATCH_SIZE,
    'n_states': N_STATES,
    'predict_mask': True,
    'reuse_weights_for_reconstruction': False,
}] * N_GPUS

loss_params = [{
    'targets' : [],
    'agg_func' : modelsource.parallel_reduce_mean,
    'loss_per_case_func' : modelsource.flex_2loss_normed,
    'loss_per_case_func_params' : {'_outputs': 'outputs', '_targets_$all': 'inputs'},
    'loss_func_kwargs' : {'gpu_id': 0, 'min_particle_distance': min_particle_distance}, 
    #{'l2_coef' : L2_COEF}
}] * N_GPUS

learning_rate_params = [{
    'func': tf.train.exponential_decay,
    'learning_rate': LRS[0],
    'decay_rate': 0.95,
    'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
    'staircase': True
}] * N_GPUS

optimizer_params = [{
    'func': modelsource.ParallelClipOptimizer,
    'optimizer_class': tf.train.RMSPropOptimizer, #tf.train.AdamOptimizer,
    'clip': True,
    'gpu_offset': 0,
    #'momentum': .9,
    #'beta1': 0.95,
    #'beta2': 0.9995,
}] * N_GPUS

validation_params = [{
    'valid0' : {
        'data_params' : {
            'func' : ShortLongSequenceDataProvider,
            'data_path' : VALDATA_PATH,
            'short_sources' : [], #'depths2', 'normals2', 'images'
            'long_sources' : ['actions', #'depths', 'objects', 
                    'object_data', 'reference_ids', 'max_coordinates', \
                    'min_coordinates', 'full_particles', 'sparse_shape_32', 
                    'is_moving', 'is_object_in_view'],
            'short_len' : SHORT_LEN,
            'long_len' : LONG_LEN,
            'min_len' : MIN_LEN,
            'filters' : None, #['is_moving', 'is_object_in_view'],
            'shuffle' : True,
            'shuffle_seed' : SEED,
            'n_threads' : 1,
            'batch_size' : DATA_BATCH_SIZE,
            'file_grab_func' : table_norot_grab_func,
           # 'is_there_subsetting_rule' : 'just_first',
            #'is_in_view_subsetting_rule' : 'first_there',
            },
        'queue_params' : {
            'queue_type' : 'fifo',
            'batch_size' : MODEL_BATCH_SIZE,
            'seed' : SEED,
            'capacity' : MODEL_BATCH_SIZE * 11,
            'min_after_dequeue': MODEL_BATCH_SIZE * 10
            },
        'targets' : {
            'func' : grab_all,
            'targets' : [],
            'num_to_save' : 5,
            'gpu_id': 0,
            'bin_file': BIN_FILE,
            },
        # 'agg_func' : lambda val_res : mean_losses_subselect_rest(val_res, 1),
        'agg_func' : keep_all,
        #'online_agg_func' : append_it,
        'num_steps' : 1, #np.round(256.0 * 12 * 4 / MODEL_BATCH_SIZE).astype(np.int32),
    },
}] * N_GPUS

train_params =  {
    'validate_first': False,
    'data_params' : {
        'func' : ShortLongSequenceDataProvider,
        'data_path' : DATA_PATH,
        'short_sources' : [], #'depths2', 'normals2', 'images' 
        'long_sources' : ['actions', #'depths', 'objects', 
                'object_data', 'reference_ids', 'max_coordinates', 'min_coordinates', \
                'full_particles', 'sparse_shape_32'],
        'short_len' : SHORT_LEN,
        'long_len' : LONG_LEN,
        'min_len' : MIN_LEN,
        'filters' : None, #['is_moving', 'is_object_in_view'],
        'shuffle' : True,
        'shuffle_seed' : SEED,
        'n_threads' : 1,
        'batch_size' : DATA_BATCH_SIZE,
        'file_grab_func' : table_norot_grab_func,
        #'is_there_subsetting_rule' : 'just_first',
        #'is_in_view_subsetting_rule' : 'first_there',
    },
        
    'queue_params' : {
        'queue_type' : 'random',
        'batch_size' : MODEL_BATCH_SIZE,
        'seed' : SEED,
        'capacity' : MODEL_BATCH_SIZE * 60
    },
    'num_steps' : float('inf'),
    'thres_loss' : float('inf')
}

for i, _ in enumerate(save_params):
    save_params[i] = copy.deepcopy(save_params[i])
    load_params[i] = copy.deepcopy(load_params[i])
    model_params[i] = copy.deepcopy(model_params[i])
    loss_params[i] = copy.deepcopy(loss_params[i])
    learning_rate_params[i] = copy.deepcopy(learning_rate_params[i])
    optimizer_params[i] = copy.deepcopy(optimizer_params[i])
    validation_params[i] = copy.deepcopy(validation_params[i])

for i, _ in enumerate(model_params):
    save_params[i]['exp_id'] = EXP_ID[i]
    load_params[i]['exp_id'] = EXP_ID[i]

    save_params[i]['cache_dir'] = CACHE_DIRS[i]
    loss_params[i]['loss_func_kwargs']['gpu_id'] = i
    #loss_params[i]['loss_func_kwargs']['bin_data_file'] = BIN_PATH + EXP_ID[i] + '.pkl'
    model_params[i]['gpu_id'] = i
    optimizer_params[i]['gpu_offset'] = i
    validation_params[i]['valid0']['targets']['gpu_id'] = i
    #validation_params[i]['valid0']['targets']['bin_file'] = BIN_PATH + EXP_ID[i] + '.pkl'
    model_params[i]['cfg'] = CFG[i]
    learning_rate_params[i]['learning_rate'] = LRS[i]

params = {
    'save_params' : save_params[0],
    'load_params' : load_params[0],
    'model_params' : model_params[0],
    #'train_params' : train_params,
    #'loss_params' : loss_params,
    #'learning_rate_params' : learning_rate_params,
    #'optimizer_params': optimizer_params,
    'validation_params' : validation_params[0],
    'inter_op_parallelism_threads': 500,
    'dont_run': True,
}

def normalize_grid(unnormalized_grid, stats):
    grid = copy.deepcopy(unnormalized_grid)
    dim = grid.shape[-1]
    grid[:,:,:,0:7] = (grid[:,:,:,0:7] - \
            stats['full_particles']['min'][0,0:7]) / \
            (stats['full_particles']['max'][0,0:7] - \
            stats['full_particles']['min'][0,0:7])
    if dim > 7:
        grid[:,:,:,7:10] = (grid[:,:,:,7:10] - \
                stats['actions']['min'][0,0:3]) / \
                (stats['actions']['max'][0,0:3] - \
                stats['actions']['min'][0,0:3])
    if dim > 10:
        grid[:,:,:,10:13] = (grid[:,:,:,10:13] - \
                stats['actions']['min'][0,3:6]) / \
                (stats['actions']['max'][0,3:6] - \
                stats['actions']['min'][0,3:6])
    if dim > 15:
        grid[:,:,:,15:18] = (grid[:,:,:,15:18] - \
                stats['full_particles']['min'][0,15:18]) / \
                (stats['full_particles']['max'][0,15:18] - \
                stats['full_particles']['min'][0,15:18])
    return grid

def unnormalize_grid(grid, stats):
    unnormalized_grid = copy.deepcopy(grid)
    dim = unnormalized_grid.shape[-1]
    unnormalized_grid[:,:,:,0:7] = unnormalized_grid[:,:,:,0:7] * \
            (stats['full_particles']['max'][0,0:7] - \
            stats['full_particles']['min'][0,0:7]) + \
            stats['full_particles']['min'][0,0:7]
    if dim > 7:
        unnormalized_grid[:,:,:,7:10] = unnormalized_grid[:,:,:,7:10] * \
                (stats['actions']['max'][0,0:3] - \
                stats['actions']['min'][0,0:3]) + \
                stats['actions']['min'][0,0:3]
    if dim > 10:
        unnormalized_grid[:,:,:,10:13] = unnormalized_grid[:,:,:,10:13] * \
                (stats['actions']['max'][0,3:6] - \
                stats['actions']['min'][0,3:6]) + \
                stats['actions']['min'][0,3:6]
    if dim > 15:
        unnormalized_grid[:,:,:,15:18] = unnormalized_grid[:,:,:,15:18] * \
                (stats['full_particles']['max'][0,15:18] - \
                stats['full_particles']['min'][0,15:18]) + \
                stats['full_particles']['min'][0,15:18]
    return unnormalized_grid

def normalize_prediction(unnormalized_pred, stats):
    pred = copy.deepcopy(unnormalized_pred)
    pred = (pred - \
            stats['full_particles']['min'][0,15:18]) / \
            (stats['full_particles']['max'][0,15:18] - \
            stats['full_particles']['min'][0,15:18])
    return pred

def unnormalize_prediction(pred, stats):
    unnormalized_pred = copy.deepcopy(pred)
    unnormalized_pred = unnormalized_pred * \
            (stats['full_particles']['max'][0,15:18] - \
            stats['full_particles']['min'][0,15:18]) + \
            stats['full_particles']['min'][0,15:18]
    return unnormalized_pred

def get_sequence_ranges(use_frame):
    isone = np.concatenate(([0], np.equal(use_frame, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isone))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def assign_actions(grid, action):
    # reset actions
    grid[:,:,:,7:13] = 0
    for force_torque, action_id in zip(action[:,0:6], action[:,8]):
        if action_id not in [-1, 0]:
            x,y,z = np.where(np.equal(grid[:,:,:,14], action_id))
            grid[x,y,z,7:13] = force_torque # actions
    # Treat special cases of particles that where melted together
    max_id = 24
    assert np.max(action[:,8]) <= max_id, '%d > max_id = %d. Check your data!' % (np.max(action[:,8]), max_id)
    for action_id in action[:,8]:
        assert action_id in [-1, 0, 23, 24]
    x,y,z = np.where(np.greater(grid[:,:,:,14], max_id))
    if len(x) > 0:
        #xt,yt,zt = grid[:,:,:,14].nonzero()
        #print('Fused particles / All particles', len(x) * 100.0 / len(xt))
        mixed_ids = grid[x,y,z,14]
        partial_ids = grid[x,y,z,13] - 1
        counts = grid[x,y,z,18]
        # determine which id the partial_id ratio refers to
        major23 = np.abs(np.round(counts * partial_ids * 23 + counts * (1 - partial_ids) * 24) - mixed_ids)
        major24 = np.abs(np.round(counts * partial_ids * 24 + counts * (1 - partial_ids) * 23) - mixed_ids)
        assert np.sum(major23) == 0 or np.sum(major24) == 0
        major = np.argmin(np.stack([major23, major24], axis=-1), axis=-1)
        major += 23 # add one if major24 otherwise major23 and add zero
        # major is a matrix where each entry indicates which key goes first 23 or 24
        n = np.zeros(major.shape)
        n[major == 23] = partial_ids[major == 23]
        n = {23: n}
        n[23][major != 23] = (1 - partial_ids[major != 23])
        n[24] = 1 - n[23]
        n[23] *= counts
        n[24] *= counts
        partial_actions = np.zeros(list(major.shape) + [6])
        for force_torque, action_id in zip(action[:,0:6], action[:,8]):
            if action_id in [23, 24]:
                partial_actions += n[action_id][:,np.newaxis] * force_torque
        grid[x,y,z,7:13] = partial_actions / counts[:,np.newaxis]
    return grid

def update_particle_position(grid):
    x,y,z = grid[:,:,:,14].nonzero()
    indices = grid[x,y,z,0:3]
    states = grid[x,y,z]
    states[:,13] -= 1
    max_coord = np.amax(indices, axis=0)
    min_coord = np.amin(indices, axis=0)
    indices = (indices - min_coord[np.newaxis,:]) / (max_coord[np.newaxis,:] - min_coord[np.newaxis,:])
    grid_dim = np.array(grid.shape[0:3])
    coordinates = np.round(indices * (grid_dim - 1)).astype(np.int32)
    coordinates = [tuple(coordinate) for coordinate in coordinates]
    sorted_coordinates_indices = sorted(range(len(coordinates)), key=coordinates.__getitem__)
    sorted_coordinates = [coordinates[i] for i in sorted_coordinates_indices]
    unique_coordinates, idx_start, counts = np.unique(sorted_coordinates, return_index=True, return_counts=True, axis=0)
    identical_coordinates_sets = np.split(sorted_coordinates_indices, idx_start[1:])
    # Weighted sum of particles depending on the number of particles in each voxel (by mass or count)
    #TODO Inelastic collision or count??? which one works better / is correct?
    counts = np.array([np.sum(states[coordinate, 18]) for coordinate in identical_coordinates_sets])[:,np.newaxis]
    mass = np.array([np.sum(states[coordinate, 3:4], axis=0) \
            for i, coordinate in enumerate(identical_coordinates_sets)])
    pos = np.array([np.sum(states[coordinate, 0:3] * states[coordinate, 18:19], axis=0) / counts[i] #mass \
            for i, coordinate in enumerate(identical_coordinates_sets)])
    vel = np.array([np.sum(states[coordinate, 4:7] * states[coordinate, 18:19], axis=0) / counts[i] #mass \
            for i, coordinate in enumerate(identical_coordinates_sets)])
    force_torque = np.array([np.sum(states[coordinate, 7:13] * states[coordinate, 18:19], axis=0) / counts[i] #mass \
            for i, coordinate in enumerate(identical_coordinates_sets)])
    partial_ids = np.array([np.sum(states[coordinate, 13:14] * states[coordinate, 18:19], axis=0) / counts[i] \
            for i, coordinate in enumerate(identical_coordinates_sets)]) + 1
    ids = np.array([np.sum(states[coordinate, 14:15], axis=0) \
            for i, coordinate in enumerate(identical_coordinates_sets)])
    next_vel = np.array([np.sum(states[coordinate, 15:18] * states[coordinate, 18:19], axis=0) / counts[i] #mass \
            for i, coordinate in enumerate(identical_coordinates_sets)])
    states = np.concatenate([pos, mass, vel, force_torque, partial_ids, ids, next_vel, counts], axis=-1)

    print('Number of particles: %d' % len(states))
    grid = np.zeros(grid.shape).astype(np.float32)
    grid[unique_coordinates[:,0], \
         unique_coordinates[:,1], \
         unique_coordinates[:,2]] = states
    return grid

if __name__ == '__main__':
    # load stats file
    f = open(STATS_FILE)
    stats = cPickle.load(f)
    # get session and data
    print('Creating Graph')
    base.get_params()
    sess, queues, dbinterface, valid_targets_dict = base.test_from_params(**params)
    # start queue runners
    print('Starting queue runners')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    # get handles to network parts
    # data ops
    get_grids = valid_targets_dict['valid0']['targets']['full_grids']
    get_actions = valid_targets_dict['valid0']['targets']['actions']
    get_is_moving = valid_targets_dict['valid0']['targets']['is_moving']
    get_in_view = valid_targets_dict['valid0']['targets']['in_view']
    get_true_velocity = valid_targets_dict['valid0']['targets']['next_velocity']
    # run model ops
    predict_velocity = valid_targets_dict['valid0']['targets']['pred_velocity']
    predict_mask = valid_targets_dict['valid0']['targets']['pred_mask']
    predict_grid = valid_targets_dict['valid0']['targets']['pred_grid']
    get_next_grid = valid_targets_dict['valid0']['targets']['next_grid']
    # placeholders
    grid_placeholder = valid_targets_dict['valid0']['targets']['grid_placeholder']
    grids_placeholder = valid_targets_dict['valid0']['targets']['grids_placeholder']
    action_placeholder = valid_targets_dict['valid0']['targets']['action_placeholder']
    # unroll across time
    print('Starting prediction')
    grids, actions, is_moving, in_view = sess.run([get_grids, get_actions, get_is_moving, get_in_view])
    grids = np.squeeze(grids)
    actions = np.squeeze(actions)
    is_moving = np.squeeze(is_moving)
    in_view = np.squeeze(in_view)
    # determine range of test sequences
    use_frame = is_moving * np.min(in_view, axis=-1)
    ranges = get_sequence_ranges(use_frame)

    # run model and reuse output as input
    predicted_sequences = []
    for i, r in enumerate(ranges):
        print('Processing sequence %d' % i)
        init = True
        predicted_grids = []
        max_coordinates = []
        min_coordinates = []
        for frame in range(16, 24): #trange(r[0], r[1]): #trange
            if init:
                grid = grids[frame]
                action = actions[frame]
                init = False

                if USE_GROUND_TRUTH_FOR_VALIDATION:
                    predicted_velocity = grids[frame,:,:,:,15:18]
                    predicted_grid = grids[frame+1,:,:,:,0:7]
                else:
                    # predict next velocity
                    predicted_velocity, predicted_grid, predicted_mask, next_grid = \
                            sess.run([predict_velocity, predict_grid, 
                                predict_mask, get_next_grid], 
                            feed_dict={
                                grid_placeholder: grid[np.newaxis,:,:,:,0:10],
                                grids_placeholder: grids[np.newaxis,frame:frame+2],
                                action_placeholder: actions[np.newaxis,frame:frame+2],
                                })
                    predicted_velocity = np.squeeze(predicted_velocity[0])
                    predicted_grid = np.squeeze(predicted_grid[0])
                    predicted_mask = np.squeeze(predicted_mask[0])
                    '''
                    res = {
                            'predicted_velocity': predicted_velocity,
                            'predicted_grid': predicted_grid,
                            'next_grid': next_grid,
                            'grids': grids, 
                            }
                    res_file = os.path.join(SAVE_DIR, 'step1_' + EXP_ID[0] + '.pkl')
                    with open(res_file, 'w') as f:
                        cPickle.dump(res, f)
                    raise NotImplementedError
                    '''

                # Undo normalization before storing
                unnormalized_grid = unnormalize_grid(grid, stats)
                predicted_grids.append(unnormalized_grid)
                x,y,z = unnormalized_grid[:,:,:,14].nonzero()
                max_coordinates.append(np.amax(unnormalized_grid[x,y,z,0:3], axis=0))
                min_coordinates.append(np.amin(unnormalized_grid[x,y,z,0:3], axis=0))
            else:
                do_move_particles = False
                if do_move_particles:
                    # Undo normalization
                    grid = unnormalize_grid(grid, stats)
                    predicted_velocity = unnormalize_prediction(predicted_velocity, 
                            stats)

                    # TODO Match particles and predictions up or not?
                    x,y,z = grid[:,:,:,14].nonzero()
                    #xp,yp,zp = np.sum(np.abs(predicted_velocity), axis=-1).nonzero()
                    grid[x,y,z,0:3] += predicted_velocity[x,y,z] # pos # mass remains unchanged
                    grid[x,y,z,4:7] = predicted_velocity[x,y,z] # velocity

                    # Assign current actions by id
                    grid = assign_actions(grid, actions[frame])

                    # Move particles to new coordinates and fuse if necessary, remember max and min coordinates or derive later
                    grid = update_particle_position(grid)
                else:
                    predicted_grid = np.concatenate([predicted_grid,
                            grids[frame,:,:,:,7:19]], axis=-1)
                    assert predicted_grid.shape[-1] == 19, predicted_grid.shape
                    predicted_grid = unnormalize_grid(predicted_grid, stats)
                    # softmax
                    predicted_mask = np.exp(predicted_mask) / \
                            np.sum(np.exp(predicted_mask), axis=0)
                    predicted_mask = np.argmax(predicted_mask, axis=-1)
                    #x,y,z = predicted_grid[:,:,:,14].nonzero()
                    x,y,z = predicted_mask.nonzero()
                    grid = np.zeros(predicted_grid.shape)
                    grid[x,y,z] = predicted_grid[x,y,z]
                    grid[x,y,z,14] = 23

                # Store unnormalized grid
                predicted_grids.append(grid.astype(np.float32))
                x,y,z = grid[:,:,:,14].nonzero()
                max_coordinates.append(np.amax(
                    grid[x,y,z,0:3], axis=0).astype(np.float32))
                min_coordinates.append(np.amin(
                    grid[x,y,z,0:3], axis=0).astype(np.float32))

                # redo normalization
                grid = normalize_grid(grid, stats)


                if USE_GROUND_TRUTH_FOR_VALIDATION:
                    predicted_velocity = grids[frame,:,:,:,15:18]
                    predicted_grid = grids[frame+1,:,:,:,0:7]
                else:
                    # predict next velocity
                    predicted_velocity, predicted_grid, predicted_mask = \
                            sess.run([predict_velocity, 
                                predict_grid, predict_mask], 
                            feed_dict={
                                grid_placeholder: grid[np.newaxis,:,:,:,0:10],
                                grids_placeholder: grids[np.newaxis,frame:frame+2],
                                action_placeholder: actions[np.newaxis,frame:frame+2],
                                })
                    predicted_velocity = np.squeeze(predicted_velocity[0])
                    predicted_grid = np.squeeze(predicted_grid[0])
                    predicted_mask = np.squeeze(predicted_mask[0])
        predicted_sequences.append({
            'predicted_grids': np.stack(predicted_grids, axis=0), 
            'max_coordinates': np.stack(max_coordinates, axis=0), 
            'min_coordinates': np.stack(min_coordinates, axis=0)})

    # Store results and ground truth in .pkl file
    print('Storing results')
    if USE_GROUND_TRUTH_FOR_VALIDATION:
        results_file = os.path.join(SAVE_DIR, 'true_results_' + EXP_ID[0] + '.pkl')
        with open(results_file, 'w') as f:
            cPickle.dump(predicted_sequences, f)
    else:
        results_file = os.path.join(SAVE_DIR, 'results_' + EXP_ID[0] + '.pkl')
        with open(results_file, 'w') as f:
            cPickle.dump(predicted_sequences, f)
    print('Results stored in ' + results_file)
