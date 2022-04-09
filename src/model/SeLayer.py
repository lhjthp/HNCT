import torch.nn as nn
import torch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)

# x = torch.randn((1,50,64,64))
# model = SELayer(50)
# total_params = sum(p.numel() for p in model.parameters())
# print("total_params", total_params)
# total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("total_trainable_params", total_trainable_params)
# print(model(x).shape)

