from torch import nn, optim
import torch,os
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义保存权重地址
weight_path = "params/unet.pth"
# 定义数据地址
data_path = r"D:\Code\python\DeepLearning\study\Unet\VOC2007"
# 定义保存训练过程图片地址
save_image_path = "train_image"

if __name__ == '__main__':
    # 加载数据集
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)# batch是批次大小，取决于性能。shuffle是打乱数据。
    # unet模型实例化
    net = UNet()
    # 检查权重文件是否存在
    if os.path.exists(weight_path):# 存在则加入网络
        net.load_state_dict(torch.load(weight_path))
        print("successful load weight!")
    else:
        print("no weight!")
    # 创建优化器
    opt = optim.Adam(net.parameters()) # 将参数传入其中以优化
    # 定义损失函数。BCELoss适合二分类问题，常用于图像分割任务。
    loss_fun = nn.BCELoss()
    # 定义轮次
    epoch = 1
    while epoch <= 100:
        # 遍历数据加载器，获取每个批次的图像和分割图像
        for i,(image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device) # 将图像移动到指定的计算设备
            out_image = net(image)# 将图像传入网络得到输出
            train_loss = loss_fun(out_image, segment_image)# 计算损失
            opt.zero_grad() # 在每个批次训练开始前，清零优化器中的梯度，以避免累加。
            train_loss.backward()# 反向计算
            opt.step()# 根据计算出的梯度更新参数
            # 每10次打印当前损失
            if i % 10 == 0:
                print('Epoch {}, Batch {}, Loss {}'.format(epoch, i, train_loss.item()))
            # 从批次中提取第一个图像以查看训练效果
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]
            # 将三个图像进行拼接
            img = torch.stack([_image, _segment_image, _out_image], 0)
            save_image(img, save_image_path + "/train_image_{}.png".format(i))
        # 每50次保存权重
        if epoch % 50 == 0:
            torch.save(net.state_dict(), weight_path)
        epoch += 1