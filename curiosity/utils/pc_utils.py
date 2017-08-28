import numpy as np
#from lib.config import cfg
from itertools import product
from math import factorial as fac
import os 
import tensorflow as tf

def pts_arr_2_control_pts(pts_arr, ctrl_dist=4, n_vox=32):
    """ Given N points, for each point return ctrl_dist ** 3
        control point locations. If a point is within the voxel grid (ie,
        in the voxel 10,10,10 for a 64x64x64 voxel), will make it roughtly 
        centered in a cube grid of control points. If it is as an edge of the
        voxel grid, will still make it centered in a non-cube grid of control 
        points by leaving some coefficients as zeroes. Note that this means this does not 
        handle edge cases that elegantly; if a point is at (31.99, 31.99, 31.99), the ctrl point
        grid will be smaller than if it's at (16.0,16.0,16.0). This can be accounted for by limiting
        inputs points arr to range of 2.0-30.0

        Note: this assumes that points have been normalized according to 
              description in vox_file, ie pts_arr=(original_pts - offset)/scale*dim 
	[they are scaled to be in 32x32x32 grid]

        input:
            category: category for caching
            model_id: model_id for caching
            pts_arr: (N, 3) np array
            vox_file: file path to the corresponding voxel
                of pts_arr
            ctrl_dist: how many control points in each dimension.
                The total number of control points is therefore
                ctrl_dist ** 3.
        return:
            (ctrl_pts,b_coeff): (N, ctrl_dist ** 3, 3),(N, ctrl_dist ** 3)
                e.g. ctrl_pts[0,:,:] = [[0, 0, 0],
                                        [0, 0, 1],
                                        ...
                                        [1, 1, 1]]
         or (batched_ctrl_pts,b_coeff): (N, ctrl_dist ** 3, 4),(N, ctrl_dist ** 3)
                e.g. batched_ctrl_pts[0, :, :] = [[batch_id, 0, 0, 0],
                                                  [batch_id, 0, 0, 1],
                                                  ...
                                                  [batch_id, 1, 1, 1]]
    """
    N = pts_arr.shape[0]
    #n_vox = cfg.CONST.N_VOX
    ctrl_pts = np.zeros([N, ctrl_dist ** 3, 3],dtype=np.int32)
    b_coeffs = np.zeros([N, ctrl_dist ** 3])
    
    point_voxels = pts_arr.astype(int)
    origin_offset = int((ctrl_dist-1)/2)
    default_origin_offset = np.repeat(int((ctrl_dist-1)/2),3) 
    point_offsets = np.minimum(np.tile(default_origin_offset,[N,1]),point_voxels)#Don't go below 0
    c_grid_origins = np.minimum(point_voxels-point_offsets,n_vox-2)#Get 'origin' ctrl point for each point
    max_dists = np.minimum(np.tile(np.repeat(ctrl_dist-1,3),[N,1]),#Compute size of local ctrl point grid for , to not go above max
                           n_vox-1-c_grid_origins)
    stus = (pts_arr-c_grid_origins)/max_dists #Scale each point to 0.0-1.0 range in its local ctrl point grid 
    for p in range(N):
        #Offset origin to roughly center point in the control point grid
        c_grid_origin = np.maximum(point_voxels[p]-origin_offset,0) 
        
        count = 0
        stu = stus[p]
        max_dist = max_dists[p]
        #for offset in product(range(ctrl_dist),repeat=3):
        for dist in product(range(ctrl_dist),repeat=3):
            if any(dist>max_dist):
              #Just leave as zeroes, won't affect anything
              continue
            ctrl_pts[p,count] = c_grid_origins[p]+dist
            
            #Compute coefficient as well
            B_ijk = 1
            dist_diff = max_dist - dist
            for i in range(3):
                b_coeff = fac(max_dist[i]) // (fac(dist[i]) * fac(dist_diff[i]))
                B_ijk*= b_coeff * (1-stus[p][i])**dist_diff[i] * stus[p][i]**dist[i]
            b_coeffs[p,count] = B_ijk
            count+=1
    return (ctrl_pts,b_coeffs)

def tf_fac(x):
    return tf.reduce_prod(tf.range(tf.maximum(x,1))+1)

def tf_pts_arr_2_control_pts(pts_arr, ctrl_dist=4, n_vox=32):
    pts_arr = tf.cast(pts_arr, tf.float32)
    B = pts_arr.get_shape().as_list()[0]
    N = pts_arr.get_shape().as_list()[1]
    ctrl_pts = tf.zeros([B, N, ctrl_dist ** 3, 3], dtype=tf.int32)
    b_coeffs = tf.zeros([B, N, ctrl_dist ** 3], dtype=tf.float32)
    
    point_voxels = tf.cast(pts_arr, tf.int32)
    origin_offset = tf.cast((ctrl_dist-1)/2, tf.int32)
    default_origin_offset = tf.tile(tf.cast((
        tf.reshape(ctrl_dist, [1])-1)/2, tf.int32), [3])
    #Don't go below 0
    point_offsets = tf.minimum(tf.tile(tf.reshape(
        default_origin_offset, [1,1,3]),
        [B,N,1]), point_voxels)
    #Get 'origin' ctrl point for each point
    c_grid_origins = tf.minimum(point_voxels-point_offsets, n_vox-2)
    #Compute size of local ctrl point grid for , to not go above max
    max_dists = tf.minimum(tf.tile(tf.reshape(tf.tile(tf.reshape(ctrl_dist, [1])-1, 
        [3]), [1,1,3]), [B,N,1]), n_vox-1-c_grid_origins)
    #Scale each point to 0.0-1.0 range in its local ctrl point grid
    stus = (pts_arr-tf.cast(c_grid_origins, tf.float32)) / tf.cast(max_dists, tf.float32)
    # get local coordinates
    dists = np.reshape(np.array([dist for dist in product(range(ctrl_dist), 
        repeat=3)]).astype(np.int32), [1, ctrl_dist**3, 3])
    valid_mask = tf.reduce_all(tf.less_equal(tf.expand_dims(dists, axis=1), 
            tf.expand_dims(max_dists, axis=2)), axis=-1)
    # get control point indices
    ctrl_pts = tf.expand_dims(c_grid_origins, axis=2) + tf.expand_dims(dists, axis=1)
    ctrl_pts = ctrl_pts * tf.cast(tf.expand_dims(valid_mask, axis=-1), tf.int32)
    # get b coefficients
    dists_diff = tf.expand_dims(max_dists, axis=2) - tf.expand_dims(dists, axis=1)
    fac_max_dists = tf.reshape(tf.map_fn(tf_fac, tf.reshape(max_dists, [-1])), 
            tf.shape(max_dists))
    fac_dists = tf.reshape(tf.map_fn(tf_fac, tf.reshape(dists, [-1])),
            tf.shape(dists))
    fac_dists_diff = tf.reshape(tf.map_fn(tf_fac, tf.reshape(dists_diff, [-1])),
            tf.shape(dists_diff))
    b_coeffs = tf.floor(tf.expand_dims(tf.cast(fac_max_dists, tf.float32), axis=2) / 
        tf.cast(tf.expand_dims(fac_dists, axis=1) * fac_dists_diff, tf.float32))
    b_coeffs = b_coeffs * (1 - tf.expand_dims(stus, axis=2)) ** \
            tf.cast(dists_diff, tf.float32) * tf.expand_dims(stus, axis=2) ** \
            tf.expand_dims(tf.cast(dists, tf.float32), axis=1)
    b_coeffs = tf.reduce_prod(b_coeffs, axis=-1)
    b_coeffs = b_coeffs * tf.cast(valid_mask, tf.float32)
    return (ctrl_pts, b_coeffs)

pts_arr = np.tile(np.arange(32)[:,np.newaxis],[3,3]) / 31.0 * 28.0 + 2.0
pts_arr = np.concatenate([pts_arr, pts_arr[:4]],axis=0)
N = 100
n_vox = 32
ctrl_pts, b_coeffs = pts_arr_2_control_pts(pts_arr)
# Tensorflow reimplementation
tf_ctrl_pts, tf_b_coeffs = tf_pts_arr_2_control_pts(tf.expand_dims(pts_arr, axis=0))
sess = tf.Session()
tf_ctrl_pts, tf_b_coeffs = sess.run([tf_ctrl_pts, tf_b_coeffs])
