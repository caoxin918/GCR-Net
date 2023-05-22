import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    x = x.transpose(2, 1)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(1)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _,_,num_dims = x.size()

    x = x.contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, num_dims, k)
    x = x.view(batch_size, num_points, num_dims,1).repeat(1, 1, 1, k)

    feature = torch.cat((feature - x, x), dim=2).permute(0, 1, 3, 2).contiguous()
    return feature


class EdgeConv(Module):
    def __init__(self, linear_tmp, aggr):
        super(EdgeConv, self).__init__()
        self.linear = linear_tmp
        self.aggr = aggr

    def forward(self, x, adj):
        x = get_graph_feature(x, 20, idx = adj)
        x = self.linear(x)
        if self.aggr == 'mean':
            x = x.mean(dim=2, keepdim=False)
        return x


class ResUnit(Module):
    def __init__(self,channels,first_channel):
        super(ResUnit, self).__init__()
        channels = channels
        self.num_channel = len(channels)
        in_channel = first_channel

        self.gcns = torch.nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.2)

        for i in range(self.num_channel):
            linear_tmp = nn.Linear(in_channel * 2, channels[i])
            hidden_layer = EdgeConv(linear_tmp, aggr = 'mean')
            self.gcns.append(hidden_layer)
            in_channel = channels[i]

        self.f = nn.Linear(first_channel, in_channel)

    def forward(self,x,adj):
        input = x
        for i in range(self.num_channel):
            x = x.to(torch.float32)
            x = self.gcns[i](x,adj)
            x = self.act(x)
        if input.size()[2] != x.size()[2]:
            input = self.f(input)
        x = input + x
        return x


class ResModule(Module):
    def __init__(self,module,num_unit,first_channel):
        super(ResModule, self).__init__()

        self.module = module
        self.num_unit = num_unit
        in_channel = first_channel

        self.units = torch.nn.ModuleList()
        for i in range(num_unit):
            a = ResUnit(module[i],in_channel)
            self.units.append(a)
            in_channel = module[i][-1]

        self.f = nn.Linear(first_channel, in_channel)

    def forward(self,x,adj):
        input = x
        for i in range(self.num_unit):
            x = self.units[i](x, adj)
        if input.size()[2] != x.size()[2]:
            input = self.f(input)
        x = input + x
        return x


class DeepGCN(Module):

    def __init__(self, args):
        super(DeepGCN, self).__init__()
        self.num_module = 5
        self.num_unit_eachModule = [2, 3, 2, 3, 2]

        self.in_channels = 4
        self.hidden_layer = [[[8, 16, 16], [8, 16, 16]],
                             [[32, 32, 64], [32, 32, 64], [32, 32, 64]],
                             [[64, 64, 128], [64, 64, 128]],
                             [[64, 32, 32], [64, 32, 32], [64, 32, 32]],
                             [[16, 8, 8], [16, 8, 8]]]
        self.out_channels = 1

        self.module2s = torch.nn.ModuleList()

        first_channel = 8
        linear_tmp = nn.Linear(self.in_channels * 2,first_channel)
        self.entry = EdgeConv(linear_tmp, aggr='mean')

        last_channel = self.hidden_layer[self.num_module - 1][self.num_unit_eachModule[self.num_module - 1] - 1][-1]
        linear_tmp = nn.Linear(last_channel * 2, self.out_channels)
        self.exit = EdgeConv(linear_tmp, aggr='mean')

        for module in range(self.num_module):
            a = ResModule(self.hidden_layer[module],self.num_unit_eachModule[module],first_channel)
            self.module2s.append(a)
            first_channel = self.hidden_layer[module][-1][-1]


    def forward(self,x):
        adj = knn(x[:, :, 0:3], 20)
        x = F.leaky_relu(self.entry(x,adj), negative_slope=0.2)
        x = x.to(torch.float32)
        x = self.module2s[0](x, adj)
        x1 = x.to(torch.float32)
        x = self.module2s[1](x1, adj)
        x2 = x.to(torch.float32)
        x = self.module2s[2](x2, adj)
        x3 = x.to(torch.float32)
        x = self.module2s[3](x3, adj)
        x4 = x.to(torch.float32)
        x = self.module2s[4](x4, adj)
        x5 = x.to(torch.float32)
        x = F.leaky_relu(self.exit(x5, adj), negative_slope=0.2).to(torch.float32)
        return x

