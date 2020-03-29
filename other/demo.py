# 加载库
import paddle.fluid as fluid
import numpy

# 定义输入数据
train_data=numpy.array([[1.0],[2.0],[3.0],[9.0]]).astype('float32')
y_true = numpy.array([[2.0],[4.0],[6.0],[18.0]]).astype('float32')

# 组建网络
x = fluid.data(name="x",shape=[None, 1],dtype='float32')
y = fluid.data(name="y",shape=[None, 1],dtype='float32')
# 下面的代码片段会直接为全连接层创建连接权值（W）和偏置（ bias ）两个可学习参数
y_predict = fluid.layers.fc(input=x,size=1,act=None)
# Y=X*w+bias
# 已知Y和对应的X的值，训练出w和bias

# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)

# 选择优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

# 网络参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

avg_cost_list=[]
# 开始训练，迭代100次
for i in range(10):
    outs = exe.run(
        feed={'x':train_data, 'y':y_true},
        fetch_list=[y_predict, avg_cost])
    avg_cost_list.append(outs[1])
    # print(outs)

# 输出训练结果
print(avg_cost_list)
