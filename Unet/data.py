from torch.utils.data import Dataset, DataLoader
import os
from utils import keep_image_size_open
from torchvision import transforms

transform = transforms.Compose([ # Compose 是一个将多个变换组合在一起的工具。它接受一个列表，列表中的每个变换都会按照顺序应用于输入数据。
    transforms.ToTensor() # ToTensor 是一个变换，它将输入的图像转换为 PyTorch 的张量格式，会将图像从PIL或numpy形式转化为浮点型张量，并将像素值从【0，255】转换到【0，1】
])

class MyDataset(Dataset):
    # 初始化数据集地址
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(self.path, "SegmentationClass"))
    # 返回文件名的数量
    def __len__(self):
        return len(self.name)
    # 获取对象中的元素
    def __getitem__(self, index):
        segment_name = self.name[index] # xxx.png
        segment_path = os.path.join(self.path, "SegmentationClass", segment_name) # 分割图像的地址
        image_path = os.path.join(self.path, "JPEGImages", segment_name.replace(".png", ".jpg")) # 原图的地址
        segment_image = keep_image_size_open(segment_path) # 处理分割图像
        image = keep_image_size_open(image_path) # 处理原图
        return transform(image), transform(segment_image) # 返回变换过后的两个图像

if __name__ == '__main__':
    dataset = MyDataset(r"D:\Code\python\DeepLearning\study\Unet\VOC2007")
    print(dataset[0][0].shape) # dataset[0]是元组（image，segment_image）
    print(dataset[0][1].shape)