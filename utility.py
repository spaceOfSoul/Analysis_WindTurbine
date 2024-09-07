from torch import nn
import numpy as np

def weights_init(m):
    """ Initialize the weights of some layers of neural networks, here Conv2D, BatchNorm, GRU, Linear
        Based on the work of Xavier Glorot
    Args:
        m: the model to initialize
    """
    classname = m.__class__.__name__
    # print("classname: ", classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('GRU') != -1:
        for weight in m.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()