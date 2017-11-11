'''Dumb meta management for system switching.'''



import cPickle
import os







#where to find it right now. dumps the new stuff here with new filenames.
current_meta_path = '/media/data4/nhaber/one_room_dataset/maxdist'

#the meta files to change
current_metas = ['val_diffobj_all_meta.pkl']
#the old path prefix in metas for replace
old_path_prefix = '/media/data4/nhaber/one_room_dataset'
#new path prefix for replace
new_path_prefix = '/data/nhaber/one_room_dataset'
#what to call them now (can be overwrite)
new_filenames = ['val_diffobj_all_meta_cluster.pkl']


full_meta_paths = [os.path.join(current_meta_path, fn) for fn in current_metas]
new_full_meta_paths = [os.path.join(current_meta_path, fn) for fn in new_filenames]

for old_fn, new_fn in zip(full_meta_paths, new_full_meta_paths):
    with open(old_fn) as stream:
        meta = cPickle.load(stream)
    meta['filenames'] = [dat_fn.replace(old_path_prefix, new_path_prefix) for dat_fn in meta['filenames']]
    with open(new_fn, 'w') as stream:
        cPickle.dump(meta, stream)











