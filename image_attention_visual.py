from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
from scipy import interpolate
from scipy.ndimage import zoom

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['font.family'] = 'sans-serif'  # 用来正常显示中文标签


# def resize_attention_weights(attention_weights, target_shape):
#     x = np.linspace(0, 1, attention_weights.shape[1])
#     y = np.linspace(0, 1, attention_weights.shape[0])
#     f = interpolate.interp2d(x, y, attention_weights, kind='linear')
#
#     target_x = np.linspace(0, 1, target_shape[1])
#     target_y = np.linspace(0, 1, target_shape[0])
#
#     resized_attention_weights = f(target_x, target_y)
#     return resized_attention_weights

def resize_attention_weights(attention_weight, new_shape):
    # 调整注意力权重的大小以匹配图像大小
    return zoom(attention_weight, (new_shape[0]/attention_weight.shape[0], new_shape[1]/attention_weight.shape[1]))



def img_attention_visualization(id, attention_weight):
    file_path = '/Users/rayss/Public/读研经历/论文/ironyDetection/imageVector2/' + id + ".jpg"
    image = Image.open(file_path)

    # 加载原始图像数组
    image_array = np.array(image)

    # 将注意力权重归一化到 [0, 1]
    attention_weight = (attention_weight - attention_weight.min()) / (attention_weight.max() - attention_weight.min())

    # 将注意力权重调整到与图像相匹配
    attention_weight = resize_attention_weights(attention_weight, image_array.shape[:2])

    # 扩展注意力权重为与图像相同的通道数
    attention_weight = np.expand_dims(attention_weight, axis=-1)

    # 将注意力权重融合到图像中
    attention_image = image_array * attention_weight

    # 可视化带有注意力效果的图像
    plt.imshow(attention_image.astype('uint8'))
    plt.axis('off')
    plt.title('图片Attention可视化')
    # 图片命名规则：id_img(img).jpg
    plt.savefig('input/prepared/' + str(id) + '_img.jpg', dpi=300)
    plt.show()
    plt.close()
