import torch
from torch import nn, Tensor


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU6(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.norm(self.conv(x)))


def get_link(layer, base_ch, growth_rate):
    if layer == 0:
        return base_ch, 0, []

    link = []
    out_channels = growth_rate

    for i in range(10):
        dv = 2 ** i
        if layer % dv == 0:
            link.append(layer - dv)

            if i > 0: out_channels *= 1.7

    out_channels = int((out_channels + 1) / 2) * 2
    in_channels = 0

    for i in link:
        ch, _, _ = get_link(i, base_ch, growth_rate)
        in_channels += ch

    return out_channels, in_channels, link


class HarDBlock(nn.Module):
    def __init__(self, c1, growth_rate, n_layers):
        super().__init__()
        self.links = []
        layers = []
        self.out_channels = 0

        for i in range(n_layers):
            out_ch, in_ch, link = get_link(i+1, c1, growth_rate)
            self.links.append(link)

            layers.append(ConvModule(in_ch, out_ch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += out_ch

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        layers = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []

            for i in link:
                tin.append(layers[i])

            if len(tin) > 1:
                x = torch.cat(tin, dim=1)
            else:
                x = tin[0]

            out = self.layers[layer](x)
            layers.append(out)

        t = len(layers)
        outs = []
        for i in range(t):
            if (i == t - 1) or (i % 2 == 1):
                outs.append(layers[i])

        out = torch.cat(outs, dim=1)
        return out