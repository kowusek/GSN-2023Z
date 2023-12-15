import torch.nn as nn
import torch

class _conv_node(nn.ModuleDict):
    def __init__(self, input_channels, output_channels, kernel_size, padding, b_norm):
        super(_conv_node, self).__init__()
        self.b_norm = b_norm
        self.add_module('conv', nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.add_module('relu', nn.ReLU(inplace=True))
        if b_norm:
            self.add_module('norm', nn.BatchNorm2d(output_channels))
        else:
            nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        if self.b_norm:
            return self.norm(self.relu(self.conv(input)))
        else:
            return self.relu(self.conv(input))

class _dense_block(nn.ModuleDict):
    def __init__(self, num_layers, channels, growth_rate, kernel_size, padding, b_norm):
        super(_dense_block, self).__init__()
        for i in range(num_layers):
            layer = _conv_node(
                channels + growth_rate * i,
                growth_rate,
                kernel_size,
                padding,
                b_norm
            )
            self.add_module('conv%d' % i, layer)
        self.add_module('last_conv', nn.Conv2d(channels + num_layers * growth_rate, channels, kernel_size=1, padding=0, bias=False))

    def forward(self, input):
        features = input
        for name, layer in self.items():
            if name[:4] == "conv":
                new_features = layer(features)
                features = torch.cat([features, new_features], 1)
        features = self.last_conv(features)
        return features

class _global_dense_block(nn.ModuleDict):
    def __init__(self, num_dense_blocks, num_layers, channels, growth_rate, kernel_size, padding, b_norm):
        super(_global_dense_block, self).__init__()
        for i in range(num_dense_blocks):
            rdb = _dense_block(num_layers, channels, growth_rate, kernel_size, padding, b_norm)
            self.add_module('rdb%d' % i, rdb)
        self.add_module('last_conv', nn.Conv2d(num_dense_blocks * channels, channels, kernel_size=1, padding=0, bias=False))

    def forward(self, input):
        features = input
        all_features = None
        for name, layer in self.items():
            if name[:3] == "rdb":
                features = layer(features)
                if all_features != None:
                    all_features = torch.cat([all_features, features], 1)
                else:
                    all_features = features
        all_features = self.last_conv(all_features)
        return all_features

class Net(nn.ModuleDict):
    def __init__(self, num_dense_blocks = 20, num_layers = 6, channels = 64, growth_rate = 32, kernel_size = 3, padding = 1, b_norm = False):
        super(Net, self).__init__()  
        self.add_module('conv1', nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False))
        self.add_module('conv2', nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
        self.add_module('global_dense', _global_dense_block(num_dense_blocks, num_layers, channels, growth_rate, kernel_size, padding, b_norm))
        self.add_module('conv3', nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
        self.add_module('upscale', nn.PixelShuffle(4))
        self.add_module('conv4', nn.Conv2d(channels // 16, channels, kernel_size=3, padding=1, bias=False))
        self.add_module('conv5', nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=False))

    def forward(self, input):
        input = self.conv1(input)
        features = self.conv2(input)
        features = self.global_dense(features)
        features = self.conv3(features) + input
        features = self.upscale(features)
        features = self.conv4(features)
        features = self.conv5(features)
        return features