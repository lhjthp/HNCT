
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=50, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True,
                 conv=default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1 - i / 10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input
        # theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base, 1, dim=0)
        # patch size for matching
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        # build feature pyramid
        for i in range(len(self.scale)):
            ref = input
            if self.scale[i] != 1:
                ref = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic',
                                    align_corners=True, recompute_scale_factor=True)
            # feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            # sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                            strides=[self.stride, self.stride],
                                            rates=[1, 1],
                                            padding='same')  # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            # feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            # sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            # group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k]
            # normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi
            # matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi * self.softmax_scale, dim=1)

            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1) / 4.
            y.append(yi)

        y = torch.cat(y, dim=0) + res * self.res_scale  # back to the mini-batch
        return y

# x = torch.randn((1,64,64,64))
# model = PyramidAttention()
# total_params = sum(p.numel() for p in model.parameters())
# print("total_params",total_params)
# total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("total_trainable_params",total_trainable_params)
# x = model(x)
# print(x.shape)