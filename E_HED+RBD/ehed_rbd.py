import torch
from torch import nn
from torch.nn import init

# vgg choice
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
# extend vgg choice --- follow the paper, you can change it
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])]}
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}


#******************************************* vgg16 *****************************************************
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


#*********************** feature map before sigmoid: build the connection and deconvolution**************
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out


#***************************************** extend vgg: side outputs**************************************
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, 1, 1, 1))

    def forward(self, x):
        return self.main(x)


#********************************************* fusion features*******************************************
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, scale = [], [], 1
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return vgg, feat_layers, concat_layers


class EHED_RBD(nn.Module):
    def __init__(self, base, feat_layers, concat_layers, connect, extract=[3, 8, 15, 22, 29], v2=True):
        super(EHED_RBD, self).__init__()
        self.extract = extract
        self.connect = connect
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)
        self.comb = nn.ModuleList(concat_layers)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.dataconv = nn.Conv2d(1,25,kernel_size=5,padding=2)
        self.extraconv1 = nn.Conv2d(14,64,kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.extraconv2 = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.extraconv3 = nn.Conv2d(128,1,kernel_size=3,padding=1)
        self.v2 = v2
        if v2: self.fuse = FusionLayer()

    def forward(self, x, y1, y2, y3, y4, label=None):
        prob, back, y, num = list(), list(), list(), 0
        origin = x
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                y.append(self.feat[num](x))
                num += 1
        # side output
        y.append(self.feat[num](self.pool(x)))
        for i, k in enumerate(range(len(y))):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))
        # fusion map
        if self.v2:
            # version2: learning fusion
            back.append(self.fuse(back))
        else:
            # version1: mean fusion
            back.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        for i in back: prob.append(i)
        tt = torch.cat([i for i in prob], 1)
        tt = torch.cat([tt, origin,y1,y2,y3,y4],1)
        tt = self.norm1(self.extraconv1(tt))
        tt = self.norm2(self.extraconv2(tt))
        tt = self.extraconv3(tt)
        return torch.sigmoid(tt)

# build the whole network
def build_modelv2():
    return EHED_RBD(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'])


# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = build_modelv2()
    img = torch.randn(2, 3, 224, 224)
    net = net.to(torch.device('cuda:0'))
    img = img.to(torch.device('cuda:0'))
    y1 = torch.randn(2,1,224,224)
    y1 = y1.to(torch.device('cuda:0'))

    out = net(img,y1,y1,y1,y1)
    print(out.shape)