import os
import copy

import numpy as np

import curiosity.utils.base as base
import curiosity.models.future_pred_diff_symmetric_coupled_with_below as modelsource
import curiosity.datasources.images_futures_and_actions as datasource

dbname = 'threeworld_future_pred_diff'
colname = 'test_symmetric_coupled_withbelow'
experiment_id = 'test0'
model_func = modelsource.get_model
model_func_kwargs = {"host": "18.93.3.135",
                     "port": 23044,
                     "datapath": "/data2/datasource6"}
data_func = datasource.getNextBatch
data_func_kwargs = copy.deepcopy(model_func_kwargs)
num_train_steps = 20480000
batch_size = 128
slippage = 0
SKDATA_ROOT = os.environ['SKDATA_ROOT']
CODE_ROOT = os.environ['CODE_ROOT']
cfgfile = os.path.join(CODE_ROOT,
                       'curiosity/curiosity/configs/future_pred_test0.cfg')
savedir = os.path.join(SKDATA_ROOT, 'futurepreddiffopt')
erase_earlier = 3
decaystep=1024000

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
         base_learningrate=0.1,
         loss_threshold=100000,
         decaystep=decaystep)
