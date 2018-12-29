import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from utility import init_squeezenet_param

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.squeeze_activation(self.squeeze(x))
        y = self.expand1x1_activation(self.expand1x1(x))
        z = self.expand3x3_activation(self.expand3x3(x))
        x = torch.cat((y,z), 1)
        return x


class FireDec(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(FireDec, self).__init__()
        self.expand1x1 = nn.Conv2d(inplanes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(inplanes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(expand1x1_planes + expand3x3_planes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.expand1x1_activation(self.expand1x1(x))
        z = self.expand3x3_activation(self.expand3x3(x))
        x = torch.cat((y,z), 1)
        x = self.squeeze_activation(self.squeeze(x))
        return x

class SqueezeSegNetEncoder(nn.Module):

    def __init__(self, in_channels, layer_weights, layer_biases):
        super(SqueezeSegNetEncoder, self).__init__()
        
        self.layer_weights = layer_weights
        self.layer_biases = layer_biases

        self.feature_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
        )

        self.feature_block2 = nn.Sequential(
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
        )
        self.feature_block3 = nn.Sequential(
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
        )
        self.feature_block4 = nn.Sequential(
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently form the rest
        self.classifier_conv = nn.Sequential(
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Initialize weights from pre-trained SqueezeNet 1.0 model
        param_counter = 0
        for module in self.feature_block1.children():
            if isinstance(module, nn.modules.conv.Conv2d):
                assert module.weight.size() == self.layer_weights[param_counter].size()
                module.weight.data = self.layer_weights[param_counter]
                assert module.bias.size() == self.layer_biases[param_counter].size()
                module.bias.data = self.layer_biases[param_counter]
                param_counter += 1

        for module in self.feature_block2.children():
            for module in module.children():
                if isinstance(module, nn.modules.conv.Conv2d):
                    assert module.weight.size() == self.layer_weights[param_counter].size()
                    module.weight.data = self.layer_weights[param_counter]
                    assert module.bias.size() == self.layer_biases[param_counter].size()
                    module.bias.data = self.layer_biases[param_counter]
                    param_counter += 1

        for module in self.feature_block3.children():
            for module in module.children():
                if isinstance(module, nn.modules.conv.Conv2d):
                    assert module.weight.size() == self.layer_weights[param_counter].size()
                    module.weight.data = self.layer_weights[param_counter]
                    assert module.bias.size() == self.layer_biases[param_counter].size()
                    module.bias.data = self.layer_biases[param_counter]
                    param_counter += 1 

        for module in self.feature_block4.children():
            for module in module.children():
                if isinstance(module, nn.modules.conv.Conv2d):
                    assert module.weight.size() == self.layer_weights[param_counter].size()
                    module.weight.data = self.layer_weights[param_counter]
                    assert module.bias.size() == self.layer_biases[param_counter].size()
                    module.bias.data = self.layer_biases[param_counter]
                    param_counter += 1

        for module in self.classifier_conv.children():
            if isinstance(module, nn.modules.conv.Conv2d):
                assert module.weight.size() == self.layer_weights[param_counter].size()
                module.weight.data = self.layer_weights[param_counter]
                assert module.bias.size() == self.layer_biases[param_counter].size()
                module.bias.data = self.layer_biases[param_counter]
                param_counter += 1

    def forward(self, x):
        x = self.feature_block1(x)
        dim1 = x.size()
        x, indices_1 = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
        x = self.feature_block2(x)
        dim2 = x.size()
        x, indices_2 = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
        x = self.feature_block3(x)
        dim3 = x.size()
        x, indices_3 = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
        x = self.feature_block4(x)
        x = self.classifier_conv(x)

        pool_ind = [indices_1, indices_2, indices_3]
        dim_ind = [dim1, dim2, dim3]
        return x, dim_ind, pool_ind

class SqueezeSegNetDecoder(nn.Module):

    def __init__(self, out_classes):
        super(SqueezeSegNetDecoder, self).__init__()

        self.inverse_classifier_conv = nn.Sequential(
            nn.Conv2d(1000, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.inverse_feature_block4 = nn.Sequential(
            FireDec(512, 512, 256, 256),
        )

        self.inverse_feature_block3 = nn.Sequential(
            FireDec(512, 384, 256, 256),
            FireDec(384, 384, 192, 192),
            FireDec(384, 256, 192, 192),
            FireDec(256, 256, 128, 128),
        )

        self.inverse_feature_block2 = nn.Sequential(
            FireDec(256, 128, 128, 128),
            FireDec(128, 128, 64, 64),
            FireDec(128, 96, 64, 64),
        )

        self.inverse_feature_block1 = nn.Sequential(
            nn.ConvTranspose2d(96, out_classes, kernel_size=10, stride=2, padding=1),
        )
    def forward(self, x, dim_ind, pool_ind):
        x = self.inverse_classifier_conv(x)
        x = self.inverse_feature_block4(x)
        x = F.max_unpool2d(x, pool_ind[2], kernel_size=3, stride=2, output_size=dim_ind[2])
        x = self.inverse_feature_block3(x)
        x = F.max_unpool2d(x, pool_ind[1], kernel_size=3, stride=2, output_size=dim_ind[1])
        x = self.inverse_feature_block2(x)
        x = F.max_unpool2d(x, pool_ind[0], kernel_size=3, stride=2, output_size=dim_ind[0])
        x = self.inverse_feature_block1(x)
        return x

class SqueezeSegNet(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(SqueezeSegNet, self).__init__()

        # Initialize convolutional layer weights from VGG16 model
        self.layer_weights, self.layer_biases = init_squeezenet_param(version=1.0)

        self.encoder = SqueezeSegNetEncoder(in_channels, self.layer_weights, self.layer_biases)
        self.decoder = SqueezeSegNetDecoder(out_classes)
        
    def forward(self, x):
        x_4, in_dimensions, pool_ind = self.encoder(x)
        x = self.decoder(x_4, in_dimensions, pool_ind)
        return x

class SqueezeSegNetEncoderOptimized(nn.Module):

    def __init__(self, in_channels, layer_weights, layer_biases):
        super(SqueezeSegNetEncoderOptimized, self).__init__()

        # Initialize convolutional layer weights from VGG16 model
        self.layer_weights = layer_weights
        self.layer_biases = layer_biases

        self.feature_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.feature_block2 = nn.Sequential(
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
        )

        self.feature_block3 = nn.Sequential(
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
        )

        self.feature_block4 = nn.Sequential(
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently form the rest
        self.classifier_conv = nn.Sequential(
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Initialize weights from pre-trained SqueezeNet 1.1 model
        param_counter = 0
        for module in self.feature_block1.children():
            if isinstance(module, nn.modules.conv.Conv2d):
                assert module.weight.size() == self.layer_weights[param_counter].size()
                module.weight.data = self.layer_weights[param_counter]
                assert module.bias.size() == self.layer_biases[param_counter].size()
                module.bias.data = self.layer_biases[param_counter]
                param_counter += 1

        for module in self.feature_block2.children():
            for module in module.children():
                if isinstance(module, nn.modules.conv.Conv2d):
                    assert module.weight.size() == self.layer_weights[param_counter].size()
                    module.weight.data = self.layer_weights[param_counter]
                    assert module.bias.size() == self.layer_biases[param_counter].size()
                    module.bias.data = self.layer_biases[param_counter]
                    param_counter += 1

        for module in self.feature_block3.children():
            for module in module.children():
                if isinstance(module, nn.modules.conv.Conv2d):
                    assert module.weight.size() == self.layer_weights[param_counter].size()
                    module.weight.data = self.layer_weights[param_counter]
                    assert module.bias.size() == self.layer_biases[param_counter].size()
                    module.bias.data = self.layer_biases[param_counter]
                    param_counter += 1 

        for module in self.feature_block4.children():
            for module in module.children():
                if isinstance(module, nn.modules.conv.Conv2d):
                    assert module.weight.size() == self.layer_weights[param_counter].size()
                    module.weight.data = self.layer_weights[param_counter]
                    assert module.bias.size() == self.layer_biases[param_counter].size()
                    module.bias.data = self.layer_biases[param_counter]
                    param_counter += 1

        for module in self.classifier_conv.children():
            if isinstance(module, nn.modules.conv.Conv2d):
                assert module.weight.size() == self.layer_weights[param_counter].size()
                module.weight.data = self.layer_weights[param_counter]
                assert module.bias.size() == self.layer_biases[param_counter].size()
                module.bias.data = self.layer_biases[param_counter]
                param_counter += 1

    def forward(self, x):
        x = self.feature_block1(x)
        dim1 = x.size()
        x, indices_1 = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
        x = self.feature_block2(x)
        dim2 = x.size()
        x, indices_2 = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
        x = self.feature_block3(x)
        dim3 = x.size()
        x, indices_3 = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
        x = self.feature_block4(x)
        x = self.classifier_conv(x)

        pool_ind = [indices_1, indices_2, indices_3]
        dim_ind = [dim1, dim2, dim3]
        return x, dim_ind, pool_ind

class SqueezeSegNetDecoderOptimized(nn.Module):

    def __init__(self, out_classes):
        super(SqueezeSegNetDecoderOptimized, self).__init__()

        self.inverse_classifier_conv = nn.Sequential(
            nn.Conv2d(1000, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.inverse_feature_block4 = nn.Sequential(
            FireDec(512, 512, 256, 256),
            FireDec(512, 384, 256, 256),
            FireDec(384, 384, 192, 192),
            FireDec(384, 256, 192, 192),
        )

        self.inverse_feature_block3 = nn.Sequential(
            FireDec(256, 256, 128, 128),
            FireDec(256, 128, 128, 128),
        )

        self.inverse_feature_block2 = nn.Sequential(
            FireDec(128, 128, 64, 64),
            FireDec(128, 64, 64, 64),
        )

        self.inverse_feature_block1 = nn.Sequential(
            nn.ConvTranspose2d(64, out_classes, kernel_size=6, stride=2, padding=1),
        )

    def forward(self, x, dim_ind, pool_ind):
        x = self.inverse_classifier_conv(x)
        x = self.inverse_feature_block4(x)
        x = F.max_unpool2d(x, pool_ind[2], kernel_size=3, stride=2, output_size=dim_ind[2])
        x = self.inverse_feature_block3(x)
        x = F.max_unpool2d(x, pool_ind[1], kernel_size=3, stride=2, output_size=dim_ind[1])
        x = self.inverse_feature_block2(x)
        x = F.max_unpool2d(x, pool_ind[0], kernel_size=3, stride=2, output_size=dim_ind[0])
        x = self.inverse_feature_block1(x)
        return x

class SqueezeSegNetOptimized(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(SqueezeSegNetOptimized, self).__init__()

        # Initialize convolutional layer weights from VGG16 model
        self.layer_weights, self.layer_biases = init_squeezenet_param(version=1.1)

        self.encoder = SqueezeSegNetEncoderOptimized(in_channels, self.layer_weights, self.layer_biases)
        self.decoder = SqueezeSegNetDecoderOptimized(out_classes)
        
    def forward(self, x):
        x_4, in_dimensions, pool_ind = self.encoder(x)
        x = self.decoder(x_4, in_dimensions, pool_ind)
        return x