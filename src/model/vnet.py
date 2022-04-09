from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def make_model(args, parent=False):
    return vnet(args)

class vnet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(vnet, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        self.scale = scale
        n_resblocks = args.n_resblocks
        act = nn.ReLU(True)
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        #
        # # define head module
        # self.head = conv(args.n_colors, n_feats, kernel_size)

        # m_conv_body = [
        #     common.ResBlock(
        #         conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        #     ) for _ in range(3)
        # ]

        # define body module
        depth = [8, 8, 8, 8]
        num_heads = [8, 8, 8, 8]
        window_size = 8
        self.window_size = window_size
        mlp_ratio = 2.0

        self.swinir = common.SwinIR(img_size=64, patch_size=1, in_chans=args.n_colors,
                 embed_dim=n_feats, depths=depth, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                 upscale=scale, img_range=args.rgb_range, upsampler='pixelshuffledirect')


    def forward(self, x):
        x = self.swinir(x)
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