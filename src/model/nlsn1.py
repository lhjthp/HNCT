from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return NLSN1(args)

class NLSN1(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NLSN1, self).__init__()
        n_l1_block = args.n_l1_block
        n_l2_block = args.n_l2_block
        n_l3_block = args.n_l3_block
        n_l4_block = args.n_l4_block
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        self.scale = scale
        act=nn.PReLU(num_parameters=1,init=0.25)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            Memnet1(n_l1_block, n_l2_block, n_l3_block, n_l4_block, conv,n_feats,kernel_size, bias=True, bn=False, act=act)
        ]
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = self.tail(res)
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

class Memnet4(nn.Module):
    def __init__(
            self, n_l4_block, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(num_parameters=1,init=0.25)):
        super(Memnet4, self).__init__()
        self.n_l4_block = n_l4_block
        m = []
        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        m.append(act)
        self.fe=nn.Sequential(*m)
        m = []
        m.append(conv(n_feats*n_l4_block,n_feats,1,bias=bias))
        self.lff = nn.Sequential(*m)

    def forward(self, x):
        res = x
        temp_res = []
        for i in range(self.n_l4_block):
            res = self.fe(res)
            temp_res.append(res)
        res = torch.cat(temp_res,1)
        res = self.lff(res) + x
        return res

class Memnet3_2(nn.Module):
    def __init__(
            self, n_l3_block, n_l4_block, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(num_parameters=1,init=0.25)):
        super(Memnet3_2, self).__init__()
        self.n_l3_block = n_l3_block
        #self.fe = nn.ModuleList([Memnet4(n_l4_block, conv, n_feats, kernel_size, bias=bias, bn=bn, act=act) for i in range(n_l3_block)])
        self.fe0 = Memnet4(n_l4_block, conv, n_feats, kernel_size, bias=bias, bn=bn, act=act)
        self.fe1 = Memnet4(n_l4_block, conv, n_feats, kernel_size, bias=bias, bn=bn, act=act)
        self.fe2 = Memnet4(n_l4_block, conv, n_feats, kernel_size, bias=bias, bn=bn, act=act)
        self.fe3 = Memnet4(n_l4_block, conv, n_feats, kernel_size, bias=bias, bn=bn, act=act)
        m = []
        m.append(conv(n_feats*n_l3_block,n_feats,1,bias=bias))
        self.lff = nn.Sequential(*m)

    def forward(self, x):
        res = x
        temp_res = []
        res = self.fe0(res)
        temp_res.append(res)
        res = self.fe1(res)
        temp_res.append(res)
        res = self.fe2(res)
        temp_res.append(res)
        res = self.fe3(res)
        temp_res.append(res)
        res = torch.cat(temp_res,1)
        res = self.lff(res) + x
        return res

class Memnet2(nn.Module):
    def __init__(
            self, n_l2_block, n_l3_block, n_l4_block, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(num_parameters=1,init=0.25)):
        super(Memnet2, self).__init__()
        self.n_l2_block = n_l2_block
        m = []
        m.append(Memnet3_2(n_l3_block, n_l4_block, conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        self.fe=nn.Sequential(*m)
        m = []
        m.append(conv(n_feats*n_l2_block,n_feats,1,bias=bias))
        self.lff = nn.Sequential(*m)

    def forward(self, x):
        res = x
        temp_res = []
        for i in range(self.n_l2_block):
            res = self.fe(res)
            temp_res.append(res)
        res = torch.cat(temp_res,1)
        res = self.lff(res) + x
        return res

class Memnet1(nn.Module):
    def __init__(
            self, n_l1_block, n_l2_block, n_l3_block, n_l4_block, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(num_parameters=1,init=0.25)):
        super(Memnet1, self).__init__()
        self.n_l1_block = n_l1_block
        m = []
        m.append(Memnet2(n_l2_block,n_l3_block, n_l4_block, conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        self.fe=nn.Sequential(*m)
        m = []
        m.append(conv(n_feats*n_l1_block,n_feats,1,bias=bias))
        self.lff = nn.Sequential(*m)

    def forward(self, x):
        res = x
        temp_res = []
        for i in range(self.n_l1_block):
            res = self.fe(res)
            temp_res.append(res)
        res = torch.cat(temp_res,1)
        res = self.lff(res) + x
        return res