import torch
import collections
from torch.distributions import Categorical

def write_line_to_file(s, f_path="progress.txt", new_file=False, verbose=False):
    code = "w" if new_file else "a"
    if verbose: print(s)
    with open(f_path, code, encoding='utf-8') as f:
        f.write(s)
        f.write("\n")


class UniqueDict(dict):
    def __init__(self, inp=None):
        self._no_dups = True
        if isinstance(inp, dict):
            super(UniqueDict,self).__init__(inp)
        else:
            super(UniqueDict,self).__init__()
            if isinstance(inp, (collections.Mapping, collections.Iterable)):
                si = self.__setitem__
                for k,v in inp:
                    si(k,v)
        self._no_dups = False

    def __setitem__(self, k, v):
        try:
            self.__getitem__(k)
            if self._no_dups:
                raise ValueError("duplicate key '{0}' found".format(k))
            else:
                super(UniqueDict, self).__setitem__(k, v)
        except KeyError:
            super(UniqueDict,self).__setitem__(k,v)


class UniqueList:

    def __init__(self, key_func=str):
        self._i = 0
        self._list = []
        self._set = set()
        self._key_func = key_func

    def tolist(self):
        return [v for v in self._list]

    def keys(self):
        return copy.deepcopy(self._set)

    def clear(self):
        self._i = 0
        self._list = []
        self._set = set()

    def append(self, val):
        key = self._key_func(val)
        if key not in self._set:
            self._set.add(key)
            self._list.append(val)
            return True
        return False

    def extend(self, vals):
        n_added = 0
        for item in vals:
            if self.append(item):
                n_added += 1
        return n_added

    def next(self):
        if self._i >= len(self._list):
            self._i  = 0
            raise StopIteration()
        item = self._list[self._i]
        self._i += 1
        return item

    def __getitem__(self, idx):
        return self._list[idx]

    def __contains__(self, item):
        key = self._key_func(item)
        return key in self._set

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return str(self)


class AverageMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        return self

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class RunningStatMeter(object):

    def __init__(self):
        self.avg = 0.
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0.
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        return self

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.max = max(self.max, val)
        self.min = min(self.min, val)


def sample_categorical_idx(weights):
    if len(weights) == 1: return 0
    idx = Categorical(probs=torch.FloatTensor(weights)).sample()
    return idx.item()