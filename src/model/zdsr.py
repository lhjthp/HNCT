from model import common
import numpy as np
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return ZDSR(args)

class ZDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ZDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ReResBlock(
                conv, n_feats, kernel_size, act=act) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
 
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
        m = []
        for i in range(2):
            #l = conv(n_feats, n_feats, kernel_size)
            #l.weight = nn.Parameter(torch.Tensor(np.ones([n_feats,n_feats,kernel_size,kernel_size])))
            #m.append(l)
            m.append(conv(n_feats, n_feats, kernel_size))
            m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(nn.Sigmoid())
        self.mul_body = nn.Sequential(*m)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += self.mul_body(x).mul(x)

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

