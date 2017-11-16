"""
Collection of useful online_agg_funcs
"""
import numpy as np

def append_dict(agg_res, cur_res, step):
    """
    Simply appends the values per key across batches
    """
    assert isinstance(cur_res, dict)
    if agg_res is None:
        agg_res = {k: [] for k in cur_res}
    for k, v in cur_res.items():
        agg_res[k].append(v)
    return agg_res

def mean_dict(agg_res, cur_res, step):
    """
    Means and appends the values per key across batches
    """
    assert isinstance(cur_res, dict)
    res = {}
    for k, v in cur_res.items():
        res[k] = np.mean(v)
    return append_dict(agg_res, res, step)
