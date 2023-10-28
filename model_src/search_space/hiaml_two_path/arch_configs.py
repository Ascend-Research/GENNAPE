import copy


class TwoPathNetConfig:

    def __init__(self, path1, path2,
                 path1_reduce_inds,
                 path2_reduce_inds):
        self.path1 = path1
        self.path2 = path2
        self.path1_reduce_inds = path1_reduce_inds
        self.path2_reduce_inds = path2_reduce_inds
        assert len(self.path1) > 0
        assert len(self.path2) > 0

    def __len__(self):
        return sum(len(c) for c in [self.path1, self.path2])

    def __str__(self):
        str_id = "Path1:{}\n".format(str(self.path1))
        str_id += "Path1ReduceInds:{}\n".format(sorted(list(self.path1_reduce_inds)))
        str_id += "Path2:{}\n".format(str(self.path2))
        str_id += "Path2ReduceInds:{}\n".format(sorted(list(self.path2_reduce_inds)))
        return str_id

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        return TwoPathNetConfig(copy.deepcopy(self.path1),
                                copy.deepcopy(self.path2),
                                copy.deepcopy(self.path1_reduce_inds),
                                copy.deepcopy(self.path2_reduce_inds))

    def load_state_dict(self, sd):
        self.path1 = sd.path1
        self.path2 = sd.path2
        self.path1_reduce_inds = sd.path1_reduce_inds
        self.path2_reduce_inds = sd.path2_reduce_inds
        return self

    def state_dict(self):
        return {
            "path1": [b.state_dict() for b in self.path1],
            "path2": [b.state_dict() for b in self.path2],
            "path1_reduce_inds": self.path1_reduce_inds,
            "path2_reduce_inds": self.path2_reduce_inds,
        }


class SinglePathBlockConfig:

    def __init__(self, op_names, op_types):
        self.op_names = op_names
        self.op_types = op_types
        assert len(op_names) > 0
        assert len(op_names) == len(op_types)

    def __len__(self):
        return len(self.op_names)

    def __str__(self):
        return "SPBlock({};{})".format("+".join(self.op_names),
                                       "+".join(self.op_types))

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        return SinglePathBlockConfig(copy.deepcopy(self.op_names),
                                     copy.deepcopy(self.op_types))

    def load_state_dict(self, sd):
        self.op_names = sd["op_names"]
        self.op_types = sd["op_types"]
        return self

    def state_dict(self):
        return {
            "op_names": self.op_names,
            "op_types": self.op_types,
        }
