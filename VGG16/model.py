import os  # 导入os模块，用于操作系统相关功能
import sys  # 导入sys模块，用于操作Python运行环境

sys.path.append(os.getcwd())  # 将当前工作目录添加到sys.path，方便模块导入

import torch  # 导入PyTorch主库
from torch import nn  # 从torch中导入神经网络模块
from torchsummary import summary  # 导入模型结构摘要工具


class VGG16(nn.Module):  # 定义VGG16模型，继承自nn.Module
    def __init__(self, *args, **kwargs):  # 构造函数
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        self.block1 = nn.Sequential(  # 第一块卷积层
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, padding=1
            ),  # 卷积层，输入通道1，输出通道64
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
        )

        self.block2 = nn.Sequential(  # 第二块卷积层
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
        )

        self.block3 = nn.Sequential(  # 第三块卷积层
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
        )

        self.block4 = nn.Sequential(  # 第四块卷积层
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
        )

        self.block5 = nn.Sequential(  # 第五块卷积层
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
        )

        self.block6 = nn.Sequential(  # 全连接层部分
            nn.Flatten(),  # 展平多维输入为一维
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),  # 全连接层
            nn.ReLU(),  # 激活函数
            nn.Dropout(p=0.5),  # Dropout防止过拟合
            nn.Linear(in_features=4096, out_features=4096),  # 全连接层
            nn.ReLU(),  # 激活函数
            nn.Dropout(p=0.5),  # Dropout防止过拟合
            nn.Linear(4096, 10),  # 输出层，10分类
        )

        for m in self.modules():  # 遍历所有子模块
            print(m)  # 打印模块信息
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )  # 使用Kaiming初始化权重

                if m.bias is not None:  # 如果有偏置
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0

            if isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 权重正态分布初始化

    def forward(self, x):  # 前向传播
        x = self.block1(x)  # 经过第一块
        x = self.block2(x)  # 经过第二块
        x = self.block3(x)  # 经过第三块
        x = self.block4(x)  # 经过第四块
        x = self.block5(x)  # 经过第五块
        x = self.block6(x)  # 经过全连接层

        return x  # 返回输出


if __name__ == "__main__":  # 脚本主入口
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备

    model = VGG16().to(device=device)  # 实例化模型并移动到设备

    print(model)  # 打印模型结构
    summary(model, input_size=(1, 224, 224), device=str(device))  # 打印模型摘要