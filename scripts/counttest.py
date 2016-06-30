import copy

import numpy as np

import curiosity.utils.base as base
import curiosity.models.obj_detector_feedforward as modelsource
import curiosity.datasources.images_and_counts as datasource

dbname = 'threeworld_count'
colname = 'test'
experiment_id = 'test0_longdr'
model_func = modelsource.get_model
model_func_kwargs = {"host": "18.93.3.135",
                     "port": 23044,
                     "datapath": "/data2/datasource6"}
data_func = datasource.getNextBatch
data_func_kwargs = copy.deepcopy(model_func_kwargs)
num_train_steps = 20480000
batch_size = 128
slippage = 0
cfgfile = '/home/yamins/curiosity/curiosity/configs/base_alexnet.cfg'
savedir = '/data/countopt'
erase_earlier = 3
decaystep=1024000

def corrfunc(indict, outdict):
    actual = indict['object_count_distributions']
    predicted = outdict['train_predictions']
    fracs = []
    for i in range(actual.shape[0]):
        objs = actual[i].nonzero()[0]
        frac = predicted[i][objs].sum() / np.abs(predicted[i]).sum()
        fracs.append(frac)
    return float(np.mean(fracs))


base.run(dbname,
         colname,
         experiment_id,
         model_func,
         model_func_kwargs,
         data_func,
         data_func_kwargs,
         num_train_steps,
         batch_size,
         slippage=slippage,
         cfgfile=cfgfile,
         savedir=savedir,
         erase_earlier=erase_earlier,
         base_learningrate=0.001,
         decaystep=decaystep,
         additional_metrics={'correctfrac': corrfunc})
