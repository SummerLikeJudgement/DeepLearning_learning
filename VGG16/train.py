import os
import sys

sys.path.append(os.getcwd())  # 添加上级目录到系统路径中，以便导入自定义模块

import time  # 导入time模块，用于计时训练过程
from torchvision.datasets import FashionMNIST  # 导入FashionMNIST数据集类
from torchvision import transforms  # 导入transforms模块，用于对图像进行预处理
from torch.utils.data import (
    DataLoader,
    random_split,
)  # 导入DataLoader用于批量加载数据，random_split用于划分数据集
import numpy as np  # 导入numpy库，常用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import torch  # 导入PyTorch主库
from torch import nn, optim  # 从torch中导入神经网络模块和优化器模块
import copy  # 导入copy模块，用于深拷贝模型参数
import pandas as pd  # 导入pandas库，用于数据处理和分析
from VGG_model import VGG16


def train_val_date_load():
    # 加载FashionMNIST训练集，并进行必要的预处理
    train_dataset = FashionMNIST(
        root="./data",  # 数据集存储路径
        train=True,  # 指定加载训练集
        download=True,  # 如果本地没有数据则自动下载
        transform=transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.ToTensor(),
            ]
        ),
    )

    # 按照8:2的比例将训练集划分为新的训练集和验证集
    train_date, val_data = random_split(
        train_dataset,
        [
            int(len(train_dataset) * 0.8),  # 80%作为训练集
            len(train_dataset) - int(len(train_dataset) * 0.8),  # 剩余20%作为验证集
        ],
    )

    # 构建训练集的数据加载器，设置批量大小为128，打乱数据，使用8个子进程加载数据
    train_loader = DataLoader(
        dataset=train_date, batch_size=16, shuffle=True, num_workers=1
    )

    # 构建验证集的数据加载器，设置批量大小为128，打乱数据，使用8个子进程加载数据
    val_loader = DataLoader(
        dataset=val_data, batch_size=16, shuffle=True, num_workers=1
    )

    return train_loader, val_loader  # 返回训练集和验证集的数据加载器


def train_model_process(model, train_loader, val_loader, epochs=10):
    # 训练模型的主流程，包含训练和验证过程
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 判断是否有GPU可用，否则使用CPU
    optimizer = optim.Adam(
        model.parameters(), lr=0.001
    )  # 使用Adam优化器，学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    model.to(device)  # 将模型移动到指定设备上

    best_model_wts = copy.deepcopy(model.state_dict())  # 保存最佳模型参数的副本
    best_acc = 0.0  # 初始化最佳验证准确率
    train_loss_all = []  # 用于记录每轮训练损失
    val_loss_all = []  # 用于记录每轮验证损失
    train_acc_all = []  # 用于记录每轮训练准确率
    val_acc_all = []  # 用于记录每轮验证准确率

    since = time.time()  # 记录训练开始时间

    for epoch in range(epochs):  # 遍历每一个训练轮次
        print(f"Epoch {epoch + 1}/{epochs}")  # 打印当前轮次信息

        train_loss = 0.0  # 当前轮训练损失总和
        train_correct = 0  # 当前轮训练正确样本数

        val_loss = 0.0  # 当前轮验证损失总和
        val_correct = 0  # 当前轮验证正确样本数

        train_num = 0  # 当前轮训练样本总数
        val_num = 0  # 当前轮验证样本总数

        for step, (images, labels) in enumerate(train_loader):  # 遍历训练集的每个批次
            images = images.to(device)  # 将图片数据移动到设备上
            labels = labels.to(device)  # 将标签数据移动到设备上

            model.train()  # 设置模型为训练模式

            outputs = model(images)  # 前向传播，得到模型输出

            pre_lab = torch.argmax(outputs, dim=1)  # 获取预测的类别标签

            loss = criterion(outputs, labels)  # 计算损失值

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            train_loss += loss.item() * images.size(0)  # 累加当前批次的损失
            train_correct += torch.sum(
                pre_lab == labels.data
            )  # 累加当前批次预测正确的样本数
            train_num += labels.size(0)  # 累加当前批次的样本数

            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc:{:.4f}".format(
                    epoch + 1,
                    epochs,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                    torch.sum(pre_lab == labels.data),
                )
            )

        for step, (images, labels) in enumerate(val_loader):  # 遍历验证集的每个批次
            images = images.to(device)  # 将图片数据移动到设备上
            labels = labels.to(device)  # 将标签数据移动到设备上
            model.eval()  # 设置模型为评估模式

            with torch.no_grad():  # 关闭梯度计算，提高验证速度，节省显存
                outputs = model(images)  # 前向传播，得到模型输出
                pre_lab = torch.argmax(outputs, dim=1)  # 获取预测的类别标签
                loss = criterion(outputs, labels)  # 计算损失值

                val_loss += loss.item() * images.size(0)  # 累加当前批次的损失
                val_correct += torch.sum(
                    pre_lab == labels.data
                )  # 累加当前批次预测正确的样本数
                val_num += labels.size(0)  # 累加当前批次的样本数

                print(
                    "Epoch [{/{}], Step [{}/{}], Val Loss: {:.4f}, Acc:{:.4f}".format(
                        epoch + 1,
                        epochs,
                        step + 1,
                        len(val_loader),
                        loss.item(),
                        torch.sum(pre_lab == labels.data),
                    )
                )

        train_loss_all.append(train_loss / train_num)  # 记录当前轮的平均训练损失
        val_loss_all.append(val_loss / val_num)  # 记录当前轮的平均验证损失
        train_acc = train_correct.double() / train_num  # 计算当前轮的训练准确率
        val_acc = val_correct.double() / val_num  # 计算当前轮的验证准确率
        train_acc_all.append(train_acc.item())  # 记录当前轮的训练准确率
        val_acc_all.append(val_acc.item())  # 记录当前轮的验证准确率
        print(
            f"Train Loss: {train_loss / train_num:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss / val_num:.4f}, Val Acc: {val_acc:.4f}"
        )  # 打印当前轮的损失和准确率
        if val_acc_all[-1] > best_acc:  # 如果当前验证准确率优于历史最佳
            best_acc = val_acc_all[-1]  # 更新最佳准确率
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存当前最佳模型参数

        # model.load_state_dict(best_model_wts)  # 可选：恢复最佳模型参数

    time_elapsed = time.time() - since  # 计算训练总耗时
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n"
        f"Best val Acc: {best_acc:.4f}"
    )  # 打印训练完成信息和最佳验证准确率

    torch.save(
        model.state_dict(), "./models/vgg16_net_best_model.pth"
    )  # 保存最终模型参数到文件
    train_process = pd.DataFrame(
        data={
            "epoch": range(1, epochs + 1),  # 轮次编号
            "train_loss_all": train_loss_all,  # 每轮训练损失
            "val_loss_all": val_loss_all,  # 每轮验证损失
            "train_acc_all": train_acc_all,  # 每轮训练准确率
            "val_acc_all": val_acc_all,  # 每轮验证准确率
        }
    )

    return train_process  # 返回训练过程的详细数据


def matplot_acc_loss(train_process):
    # 绘制训练和验证的损失及准确率曲线
    plt.figure(figsize=(12, 5))  # 创建一个宽12高5的画布

    plt.subplot(1, 2, 1)  # 创建1行2列的子图，激活第1个
    plt.plot(
        train_process["epoch"], train_process["train_loss_all"], label="Train Loss"
    )  # 绘制训练损失曲线
    plt.plot(
        train_process["epoch"], train_process["val_loss_all"], label="Val Loss"
    )  # 绘制验证损失曲线
    plt.xlabel("Epoch")  # 设置x轴标签为Epoch
    plt.ylabel("Loss")  # 设置y轴标签为Loss
    plt.title("Loss vs Epoch")  # 设置子图标题
    plt.legend()  # 显示图例

    plt.subplot(1, 2, 2)  # 激活第2个子图
    plt.plot(
        train_process["epoch"], train_process["train_acc_all"], label="Train Acc"
    )  # 绘制训练准确率曲线
    plt.plot(
        train_process["epoch"], train_process["val_acc_all"], label="Val Acc"
    )  # 绘制验证准确率曲线
    plt.xlabel("Epoch")  # 设置x轴标签为Epoch
    plt.ylabel("Accuracy")  # 设置y轴标签为Accuracy
    plt.title("Accuracy vs Epoch")  # 设置子图标题
    plt.legend()  # 显示图例

    plt.tight_layout()  # 自动调整子图间距
    plt.ion()  # 关闭交互模式，防止图像自动关闭
    plt.show()  # 显示所有图像
    plt.savefig("./models/vgg16_net_output.png")


if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    traindatam, valdata = train_val_date_load()  # 加载训练集和验证集
    result = train_model_process(VGG16(), traindatam, valdata, 10)
    matplot_acc_loss(result)  # 绘制训练和验证的损失及准确率曲线