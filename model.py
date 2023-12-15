import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from math import log10
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataset import get_train_set, get_val_set, get_test_set

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
        features = self.last_conv(features) + input
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


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, image_size, scaling_factor, train_dataset_path, val_dataset_path, test_dataset_path):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.scaling_factor = scaling_factor
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path

    def train_dataloader(self):
        train_set = get_train_set(self.scaling_factor,self.train_dataset_path, self.image_size)
        training_data_loader = DataLoader(
            dataset=train_set, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)
        return training_data_loader

    def val_dataloader(self):
        val_set = get_val_set(self.scaling_factor,self.val_dataset_path, self.image_size)
        val_data_loader = DataLoader(
            dataset=val_set, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)
        return val_data_loader
        
    def test_dataloader(self):
        test_set = get_test_set(self.scaling_factor,self.test_dataset_path, self.image_size)
        testing_data_loader = DataLoader(
            dataset=test_set, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)
        return testing_data_loader


class Model(pl.LightningModule):
    def __init__(self, model, scaling_factor, learning_rate):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.model = model
        self.learning_rate = learning_rate
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.train_psnr(y_hat, y)
        self.log('train_psnr', self.train_psnr, on_step=True, on_epoch=True)
        self.train_ssim(y_hat, y)
        self.log('train_ssim', self.train_ssim, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.val_psnr(y_hat, y)
        self.log('val_psnr', self.val_psnr, on_step=True, on_epoch=True)
        self.val_ssim(y_hat, y)
        self.log('val_ssim', self.val_ssim, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.test_psnr(y_hat, y)
        self.log('test_psnr', self.test_psnr, on_step=True, on_epoch=True)
        self.test_ssim(y_hat, y)
        self.log('test_ssim', self.test_ssim, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

