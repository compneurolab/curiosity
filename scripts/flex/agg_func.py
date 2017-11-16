"""
Collection of useful agg_funcs
"""
import numpy as np

def append_dict(res):
    """
    Appends all batches into a list
    """
    assert isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict)
    keys = res[0].keys()
    return dict((k, [r[k] for r in res]) for k in keys)

def concat_dict(res):
    """
    Concatenates all batches across the first dimension
    """
    assert isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict)
    res = append_dict(res)
    return dict((k, np.concatenate(res[k], axis=0)) for k in res)
