from torch import nn, cat, randn
from torch.nn import functional as F

# 卷积板块
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):# 每次卷积的输入输出不一样
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential( #定义一个顺序容器，其中包含多个神经网络层，这些层会按顺序执行。
            # 第一个卷积层
            nn.Conv2d(in_channels, out_channels, 3,1,1,padding_mode="reflect",bias=False),# 3*3的卷积核，步长为1，填充大小为1（保证输出尺寸与输入尺寸相同），反射填充，不使用偏置项
            nn.BatchNorm2d(out_channels), # 对输出进行批归一化
            nn.Dropout2d(0.3), # 随机丢弃30%神经元，减少过拟合
            nn.LeakyReLU(), # 激活函数是Leaky ReLU
            # 第二个卷积层
            nn.Conv2d(out_channels, out_channels, 3,1,1,padding_mode="reflect",bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )
    # 定义前向传播方法，接受输入x
    def forward(self, x):
        return self.layer(x)
# 下采样板块
class Downsample(nn.Module):
    def __init__(self,channels):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3,2,1,padding_mode="reflect",bias=False),# 步长为2，每次操作后尺寸将减半
            nn.BatchNorm2d(channels),#可用可不用
            nn.LeakyReLU()
        )
    # 定义前向传播方法，接受输入x
    def forward(self, x):
        return self.layer(x)
# 上采样板块
class Upsample(nn.Module):
    def __init__(self,channels):
        super(Upsample, self).__init__()
        self.layer = nn.Conv2d(channels, channels//2, 1,1)

    def forward(self, x, feature_map):# x是前序的图，feature_map是连接的特征图。
        up = F.interpolate(x, scale_factor=2, mode='nearest')# 使用最近邻插值法进行上采样，将新尺寸变为2倍。
        out = self.layer(up)
        return cat((out, feature_map), dim=1)# 将out和feature_map沿通道维度拼接在一起。
# unet板块
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1=Conv_Block(3, 64)# 输入通道为3（RGB），输出通道为64
        self.d1=Downsample(64)
        self.c2=Conv_Block(64, 128)
        self.d2=Downsample(128)
        self.c3=Conv_Block(128, 256)
        self.d3=Downsample(256)
        self.c4=Conv_Block(256, 512)
        self.d4=Downsample(512)
        self.c5=Conv_Block(512, 1024)
        self.u1=Upsample(1024)
        self.c6=Conv_Block(1024, 512)
        self.u2=Upsample(512)
        self.c7=Conv_Block(512, 256)
        self.u3=Upsample(256)
        self.c8=Conv_Block(256, 128)
        self.u4=Upsample(128)
        self.c9=Conv_Block(128, 64)
        self.out = nn.Conv2d(64, 3, 3, 1,1)
        self.Th = nn.Sigmoid() # 将输出值压缩到[0,1]内

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5,R4))# R5和R4进行拼接
        O2 = self.c7(self.u2(O1,R3))
        O3 = self.c8(self.u3(O2,R2))
        O4 = self.c9(self.u4(O3,R1))

        return self.Th(self.out(O4))

if __name__ == '__main__':
    x = randn(2,3,256,256)
    net = UNet()
    print(net(x).shape)