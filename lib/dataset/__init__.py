from ._360cc import _360CC
from ._own import _OWN
from ._cf import _CF
from ._w_not_fix import _W_Not_Fix, alignCollate
from ._w_pad import _W_Pad

def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    elif config.DATASET.DATASET == "CF":
        return _CF
    elif config.DATASET.DATASET == "W_NOT_FIX":
        return _W_Not_Fix
    elif config.DATASET.DATASET == "W_PAD":
        return _W_Pad
    else:
        raise NotImplemented()


def get_collate_fn(config):
    if config.DATASET.DATASET == "W_NOT_FIX":
        return alignCollate
    else:
        return None
