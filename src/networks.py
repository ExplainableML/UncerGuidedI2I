import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

### components
class ResConv(nn.Module):
    """
    Residual convolutional block, where
    convolutional block consists: (convolution => [BN] => ReLU) * 3
    residual connection adds the input to the output
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x_in = self.double_conv1(x)
        x1 = self.double_conv(x)
        return self.double_conv(x) + x_in

class Down(nn.Module):
    """Downscaling with maxpool then Resconv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
	"""Upscaling then double conv"""
	def __init__(self, in_channels, out_channels, bilinear=True):
		super().__init__()
		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = ResConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = ResConv(in_channels, out_channels)
	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(
			x1, 
			[
				diffX // 2, diffX - diffX // 2,
				diffY // 2, diffY - diffY // 2
			]
		)
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
	def forward(self, x):
		# return F.relu(self.conv(x))
		return self.conv(x)

##### The composite networks
class UNet(nn.Module):
	def __init__(self, n_channels, out_channels, bilinear=True):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.out_channels = out_channels
		self.bilinear = bilinear
		####
		self.inc = ResConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, out_channels)
	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		y = self.outc(x)
		return y

class CasUNet(nn.Module):
	def __init__(self, n_unet, io_channels, bilinear=True):
		super(CasUNet, self).__init__()
		self.n_unet = n_unet
		self.io_channels = io_channels
		self.bilinear = bilinear
		####
		self.unet_list = nn.ModuleList()
		for i in range(self.n_unet):
			self.unet_list.append(UNet(self.io_channels, self.io_channels, self.bilinear))
	def forward(self, x):
		y = x
		for i in range(self.n_unet):
			if i==0:
				y = self.unet_list[i](y)
			else:
				y = self.unet_list[i](y+x)
		return y

class CasUNet_2head(nn.Module):
	def __init__(self, n_unet, io_channels, bilinear=True):
		super(CasUNet_2head, self).__init__()
		self.n_unet = n_unet
		self.io_channels = io_channels
		self.bilinear = bilinear
		####
		self.unet_list = nn.ModuleList()
		for i in range(self.n_unet):
			if i != self.n_unet-1:
				self.unet_list.append(UNet(self.io_channels, self.io_channels, self.bilinear))
			else:
				self.unet_list.append(UNet_2head(self.io_channels, self.io_channels, self.bilinear))
	def forward(self, x):
		y = x
		for i in range(self.n_unet):
			if i==0:
				y = self.unet_list[i](y)
			else:
				y = self.unet_list[i](y+x)
		y_mean, y_sigma = y[0], y[1]
		return y_mean, y_sigma

class CasUNet_3head(nn.Module):
	def __init__(self, n_unet, io_channels, bilinear=True):
		super(CasUNet_3head, self).__init__()
		self.n_unet = n_unet
		self.io_channels = io_channels
		self.bilinear = bilinear
		####
		self.unet_list = nn.ModuleList()
		for i in range(self.n_unet):
			if i != self.n_unet-1:
				self.unet_list.append(UNet(self.io_channels, self.io_channels, self.bilinear))
			else:
				self.unet_list.append(UNet_3head(self.io_channels, self.io_channels, self.bilinear))
	def forward(self, x):
		y = x
		for i in range(self.n_unet):
			if i==0:
				y = self.unet_list[i](y)
			else:
				y = self.unet_list[i](y+x)
		y_mean, y_alpha, y_beta = y[0], y[1], y[2]
		return y_mean, y_alpha, y_beta

class UNet_2head(nn.Module):
	def __init__(self, n_channels, out_channels, bilinear=True):
		super(UNet_2head, self).__init__()
		self.n_channels = n_channels
		self.out_channels = out_channels
		self.bilinear = bilinear
		####
		self.inc = ResConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		#per pixel multiple channels may exist
		self.out_mean = OutConv(64, out_channels)
		#variance will always be a single number for a pixel
		self.out_var = nn.Sequential(
			OutConv(64, 128),
			OutConv(128, 1),
		)
	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		y_mean, y_var = self.out_mean(x), self.out_var(x)
		return y_mean, y_var

class UNet_3head(nn.Module):
	def __init__(self, n_channels, out_channels, bilinear=True):
		super(UNet_3head, self).__init__()
		self.n_channels = n_channels
		self.out_channels = out_channels
		self.bilinear = bilinear
		####
		self.inc = ResConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		#per pixel multiple channels may exist
		self.out_mean = OutConv(64, out_channels)
		#variance will always be a single number for a pixel
		self.out_alpha = nn.Sequential(
			OutConv(64, 128),
			OutConv(128, 1),
			nn.ReLU()
		)
		self.out_beta = nn.Sequential(
			OutConv(64, 128),
			OutConv(128, 1),
			nn.ReLU()
		)
	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		y_mean, y_alpha, y_beta = self.out_mean(x), \
		self.out_alpha(x), self.out_beta(x)
		return y_mean, y_alpha, y_beta

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [  
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features),
			nn.ReLU(inplace=True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features)
		]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial convolution block       
        model = [
			nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True)
		]
        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  
				nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True) 
			]
            in_features = out_features
            out_features = in_features*2
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  
				nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
				nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
			]
            in_features = out_features
            out_features = in_features//2
        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

### discriminator
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        """Standard forward."""
        return self.model(input)