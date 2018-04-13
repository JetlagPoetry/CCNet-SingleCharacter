import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random

# open file
try:
    with open('Xdata.txt', 'rb') as x_file:
        X_load = pickle.load(x_file)
    with open('Ydata.txt', 'rb') as y_file:
        y_load = pickle.load(y_file)
except IOError as err:
    print('File error: ' + str(err))
except pickle.PickleError as perr:
    print('Pickling error: ' + str(perr))

print(X_load.shape)
print(y_load.shape)

for i in range(10):
    img = X_load[10 * 1000 + i, :]
    img = img.reshape(48, 48)
    plt.imshow(img, cmap='gray')
    plt.show()

for i in range(4):
    for j in range(2):
        img = X_load[i * 1000 + j, :]
        img = img.reshape(48, 48)

        # 1111111111111111111旋转
        pil_image = Image.fromarray(img)
        angle = 180
        mode = Image.BICUBIC
        image1 = pil_image.rotate(angle, mode)

        # 222222222222222222翻转
        image2 = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        # 333333333333333333图像变换
        # im.transform(size, method, data) ⇒ image

        # im.transform(size, method, data, filter) ⇒ image
        # 1：image.transform((300,300), Image.EXTENT, (0, 0, 300, 300))
        #   变量data为指定输入图像中两个坐标点的4元组(x0,y0,x1,y1)。
        #   输出图像为这两个坐标点之间像素的采样结果。
        #   例如，如果输入图像的(x0,y0)为输出图像的（0，0）点，(x1,y1)则与变量size一样。
        #   这个方法可以用于在当前图像中裁剪，放大，缩小或者镜像一个任意的长方形。
        #   它比方法crop()稍慢，但是与resize操作一样快。
        # 2：image.transform((300,300), Image.AFFINE, (1, 2,3, 2, 1,4))
        #   变量data是一个6元组(a,b,c,d,e,f)，包含一个仿射变换矩阵的第一个两行。
        #   输出图像中的每一个像素（x，y），新值由输入图像的位置（ax+by+c, dx+ey+f）的像素产生，
        #   使用最接近的像素进行近似。这个方法用于原始图像的缩放、转换、旋转和裁剪。
        # 3: image.transform((300,300), Image.QUAD, (0,0,0,500,600,500,600,0))
        #   变量data是一个8元组(x0,y0,x1,y1,x2,y2,x3,y3)，它包括源四边形的左上，左下，右下和右上四个角。
        # 4: image.transform((300,300), Image.MESH, ())
        #   与QUAD类似，但是变量data是目标长方形和对应源四边形的list。
        # 5: image.transform((300,300), Image.PERSPECTIVE, (1,2,3,2,1,6,1,2))
        #   变量data是一个8元组(a,b,c,d,e,f,g,h)，包括一个透视变换的系数。
        #   对于输出图像中的每个像素点，新的值来自于输入图像的位置的(a x + b y + c)/(g x + h y + 1),
        #   (d x+ e y + f)/(g x + h y + 1)像素，使用最接近的像素进行近似。
        #   这个方法用于原始图像的2D透视。
        image3 = pil_image.transform((48, 48), Image.EXTENT, (0, 0, 100, 100))

        # 44444444444444444444444对图像进行裁剪
        crop_win_size = np.random.randint(28, 48)
        random_region = (
            (48 - crop_win_size) >> 1, (48 - crop_win_size) >> 1, (48 + crop_win_size) >> 1,
            (48 + crop_win_size) >> 1)
        image4 = pil_image.crop(random_region)

        # 55555555555555555555555对图像色彩进行抖动
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(pil_image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        image5 = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

        # 666666666666666666666666对图像进行高斯噪声处理
        mean = 0.2
        sigma = 0.3


        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im


        img_gau = gaussianNoisy(img[:, :].flatten(), mean, sigma)
        image6 = img_gau.reshape([48, 48])

        plt.imshow(image6, cmap='gray')
        plt.show()