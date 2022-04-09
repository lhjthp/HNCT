from model import common
import torch.nn as nn
def make_model(args, parent=False):
    return RERESNET(args)

class ReResNetBlock(nn.Module):
    def __init__(self, n_feats, reduction=4, conv=common.default_conv):
        super(ReResNetBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats // reduction, 3)
        self.act = nn.ReLU(True)
        self.conv2 = conv(n_feats // reduction, n_feats, 3)

    def forward(self, input):
        far_in = input[0]
        near_in = input[1]
        if far_in is not None:
            all_in = far_in + near_in
        else:
            all_in = near_in
        far_out = self.conv2(self.act(self.conv1(all_in)))
        near_out = near_in
        return near_out, far_out

class RERESNET(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RERESNET, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ReResNetBlock(n_feats, reduction) for _ in range(n_resblocks)
        ]
        self.conv_after_body = conv(n_feats, n_feats, kernel_size)

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        near_out, far_out = self.body([None, x])
        res = near_out + far_out
        res = self.conv_after_body(res) + x

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

