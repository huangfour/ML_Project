# 定义变量
import paddle.fluid as fluid
a = fluid.data(name="a", shape=[None, 1], dtype='int32')
b = fluid.data(name="b", shape=[None, 1], dtype='int32')

# 组建网络（此处网络仅由一个操作构成，即elementwise_add）
result = fluid.layers.elementwise_add(a,b)

# 准备运行网络
cpu = fluid.CPUPlace() # 定义运算设备，这里选择在CPU下训练
exe = fluid.Executor(cpu) # 创建执行器
exe.run(fluid.default_startup_program()) # 网络参数初始化

# 读取输入数据
import numpy
data_1 = 3
data_2 = 4
x = numpy.array([[data_1]])
y = numpy.array([[data_2]])


# 运行网络
outs = exe.run(
    feed={'a':x, 'b':y}, # 将输入数据x, y分别赋值给变量a，b
    fetch_list=[a,b,result] # 通过fetch_list参数指定需要获取的变量结果
    )

print(outs)
# 输出计算结果
print ("%d+%d=%d" % (data_1,data_2,outs[0][0]))