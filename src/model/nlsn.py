from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return NLSN(args)


class Memnet(nn.Module):
    def __init__(
            self, n_block, conv, n_feats, kernel_size,
            bias=True, n_color=3, act=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(Memnet, self).__init__()
        self.n_block = n_block
        self.fe = nn.ModuleList()
        for _ in range(n_block):
            m = nn.ModuleList([
                act,
                conv(n_feats, n_feats, kernel_size, bias=bias)
                ]
            )
            self.fe.append(m)
        m = []
        m.append(conv(n_feats * n_block, n_feats, 1, bias=bias))
        depth = 2
        num_heads = 2
        window_size = 8
        resolution = 64
        mlp_ratio = 2.0
        m.append(common.BasicLayer(dim=n_feats,
                                   depth=depth,
                                   resolution=resolution,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=True, qk_scale=None,
                                   norm_layer=None))


        self.lff = nn.Sequential(*m)

    def forward(self, x):
        res = x
        temp_res = []
        for i, layers in enumerate(self.fe):
            temp_res.append(res)
            for j, layer in enumerate(layers):
                res = layer(res)
        res = res + x
        up_res = torch.cat(temp_res, 1)
        up_res = self.lff(up_res)
        up_res = up_res + temp_res[0]
        return up_res, res


class nodenet(nn.Module):
    def __init__(
            self, n_block2, n_block3, conv, n_feats, kernel_size,
            bias=True, n_color=3, act=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(nodenet, self).__init__()
        self.n_block = n_block2
        self.fe = nn.ModuleList()
        for _ in range(n_block2):
            m = nn.ModuleList([
                Memnet(n_block3, conv, n_feats, kernel_size, n_color=n_color)
            ])
            self.fe.append(m)
        m = []
        m.append(conv(n_feats * n_block2, n_feats, 1, bias=bias))
        depth = 2
        num_heads = 2
        window_size = 8
        resolution = 64
        mlp_ratio = 2.0
        m.append(common.BasicLayer(dim=n_feats,
                                   depth=depth,
                                   resolution=resolution,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=True, qk_scale=None,
                                   norm_layer=None))
        self.lff = nn.Sequential(*m)


    def forward(self, x):
        res = x
        temp_res = []
        for _, layers in enumerate(self.fe):
            for _, layer in enumerate(layers):
                up_res, res = layer(res)
            temp_res.append(up_res)
        res = res + x
        up_res = torch.cat(temp_res, 1)
        up_res = self.lff(up_res)
        up_res = up_res + temp_res[0]
        return up_res, res

class rootnet(nn.Module):
    def __init__(
            self, n_block1, n_block2, n_block3, conv, n_feats, kernel_size,
            bias=True, act=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(rootnet, self).__init__()
        self.n_block1 = n_block1
        self.fe = nn.ModuleList()
        for _ in range(n_block1):
            m = nn.ModuleList([nodenet(n_block2, n_block3, conv, n_feats, kernel_size, n_color=3, act=act)])
            self.fe.append(m)
        m = []
        m.append(conv(n_feats * n_block1, n_feats, 1, bias=bias))
        depth = 2
        num_heads = 2
        window_size = 8
        resolution = 64
        mlp_ratio = 2.0
        m.append(common.BasicLayer(dim=n_feats,
                                   depth=depth,
                                   resolution=resolution,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=True, qk_scale=None,
                                   norm_layer=None))


        self.lff = nn.Sequential(*m)

    def forward(self, x):
        res = x
        temp_res = []
        for _, layers in enumerate(self.fe):
            for _, layer in enumerate(layers):
                up_res, res = layer(res)
            temp_res.append(up_res)
        res = res + x
        up_res = torch.cat(temp_res, 1)
        up_res = self.lff(up_res)
        up_res = up_res + temp_res[0]
        return up_res, res

class NLSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NLSN, self).__init__()
        n_block1 = args.n_l1_block
        n_block2 = args.n_l2_block
        n_block3 = args.n_l3_block
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        self.scale = scale
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        # act=nn.PReLU(num_parameters=1,init=0.25)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_body = [rootnet(n_block1, n_block2, n_block3, conv, n_feats, kernel_size)]
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, lr):
        lr = self.sub_mean(lr)
        x = self.head(lr)
        res, _ = self.body(x)
        res = res + x
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