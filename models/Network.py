import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .STRA import *

############################################################################
############################################################################
############################################################################


def build_net(model_name,base_channel,num_res,beta):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == 'STRA':
        return STRA(base_channel=base_channel,num_res=num_res)

    raise ModelError('Wrong Model!')
