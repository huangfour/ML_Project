import paddle.fluid as fluid
# 除 fluid.data 之外，我们还可以使用
# fluid.layers.fill_constant 来创建常量，
# 如下代码将创建一个维度为[3, 4],
# 数据类型为int64的Tensor，其中所有元素均为16（value参数所指定的值）。
data = fluid.layers.fill_constant(shape=[3, 4], value=16, dtype='int64')
print(data)
# 定义一个数据类型为int64的二维数据变量x，x第一维的维度为3，第二个维度未知，要在程序执行过程中才能确定，因此x的形状可以指定为[3, None]
x = fluid.data(name="x", shape=[3, None], dtype="int64")
print(x)
# 大多数网络都会采用batch方式进行数据组织，batch大小在定义时不确定，因此batch所在维度（通常是第一维）可以指定为None
batched_x = fluid.data(name="batched_x", shape=[None, 3, None], dtype='int64')
print(batched_x)