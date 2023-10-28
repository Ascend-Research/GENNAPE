import copy


class BlockConfig:

    def __init__(self, paths):
        # Each path is a dict with the following:
        # {"ops":[...], "op_types":[...],
        # "C_out": int,
        # "C_internal": int}
        self.paths = paths

    @staticmethod
    def _get_path_id(path):
        op_str_list = ["{}[{}]".format(op, t) for op, t in zip(path["ops"], path["op_types"])]
        return "->".join(op_str_list) + "+C[{}->{}]".format(path["C_internal"], path["C_out"])

    def __len__(self):
        return sum(len(d["ops"]) for d in self.paths)

    def __str__(self):
        path_ids = [self._get_path_id(p) for p in self.paths]
        path_ids.sort()
        return "BlockConfig:\n  " + "\n  ".join(path_ids).rstrip()

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        return BlockConfig(
            [copy.deepcopy(p) for p in self.paths]
        )

    def state_dict(self):
        return {
            "paths": self.paths
        }

    def load_state_dict(self, sd):
        self.paths = sd["paths"]
        return self


class NetConfig:

    def __init__(self, stages):
        # Each stage is a dict with the following:
        # {"block":BlockConfig, "n_blocks":int,
        # "C_in": int, "C_out": int,
        # "stride": 1 or 2,}
        self.stages = stages

    @staticmethod
    def _get_stage_id(stage, prefix):
        block_id = str(stage["block"])
        meta_str = "N:{}; C_in:{}; C_out:{}; Stride:{}".format(stage["n_blocks"],
                                                               stage["C_in"],
                                                               stage["C_out"],
                                                               stage["stride"])
        return prefix + " " + meta_str + "\n" + block_id

    def __str__(self):
        str_id = "\n"
        for si, stage in enumerate(self.stages):
            str_id += self._get_stage_id(stage, "Stage{}".format(si + 1))
            str_id += "\n"
        return str_id

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        return NetConfig(
            [{"block": copy.deepcopy(s["block"]),
              "n_blocks": s["n_blocks"],
              "C_in": s["C_in"],
              "C_out": s["C_out"],
              "stride": s["stride"]}
             for s in self.stages]
        )

    def state_dict(self):
        return {
            "stages": [{
                "block":s["block"].state_dict(),
                "n_blocks":s["n_blocks"],
                "C_in": s["C_in"],
                "C_out": s["C_out"],
                "stride": s["stride"]
            } for s in self.stages]
        }

    def load_state_dict(self, sd):
        new_stages = []
        for d in sd.stages:
            new_d = {
                "block": d["block"],
                "n_blocks": d["n_blocks"],
                "C_in": d["C_in"],
                "C_out": d["C_out"],
                "stride": d["stride"]
            }
            new_stages.append(new_d)
        self.stages = new_stages
        return self
