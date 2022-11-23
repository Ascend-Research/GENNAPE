import numpy as np

def mean(list_val, fallback_val=None):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ZeroDivisionError()
    return sum(list_val) / len(list_val)


def variance(list_val, fallback_val=0):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ValueError()
    v = np.var(np.asarray(list_val))
    return v
