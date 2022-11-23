import collections


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
