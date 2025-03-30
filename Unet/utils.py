from PIL import Image

# 对不同图片进行处理
def keep_image_size_open(path, size=(256,256)):
    img = Image.open(path)# 使用PIL库打开指定路径的图像
    temp = max(img.size) # 获取最长边
    mask = Image.new('RGB', (temp,temp), (0,0,0)) # 创建一个新图像mask
    mask.paste(img, (0,0))#将img粘贴到mask的左上角（0，0，0）
    mask = mask.resize(size)# 将mask的大小调整为size大小,正方形的缩放不会变形
    return mask