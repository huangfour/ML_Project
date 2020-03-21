from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im =np.array(Image.open("moun.jpg").convert('L'))  #灰度化
print(im.shape, im.ndim)
# 原图
plt.figure("image")
plt.imshow(im)
plt.show()
# # 变换后的图
b = im/2  #像素变换
im = Image.fromarray(b.astype('uint8'))
im.save("moun2.jpg")
