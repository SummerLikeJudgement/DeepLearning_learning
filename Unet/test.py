import os
import torch
from net import *
from utils import keep_image_size_open
from data import transform
from torchvision.utils import save_image

# 实例化网络
# net = UNet().cuda() # 有gpu
net = UNet() # 无gpu
# 导入权重到网络
weight = "params/unet.pth"
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight)) # 将权重导入到网络中
    print("load weights success")
else:
    print("load weights fail")
# 接受输入图片
_input = input("please input your image path:")
# 对输入的图片进行预处理，同data
img = keep_image_size_open(_input)
# img_data = transform(img).cuda() # 有gpu
img_data = transform(img)
img_data = torch.unsqueeze(img_data, 0)# 对img_data升维
# 将img_data传入网络得到输出
out = net(img_data)
save_image(out, "result/result.jpg")
print(out)