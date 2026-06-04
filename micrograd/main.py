from micrograd.engine import Value
from micrograd.nn import *

xs = [
    [2.0, 3.0],   # 期待输出：1.0
    [1.0, -1.0],  # 期待输出：-1.0
    [-1.0, 2.0],  # 期待输出：-1.0
    [3.0, 1.0],   # 期待输出：1.0
]
# x = Value(xs)这里很愚蠢，Value函数的输入是data，self.data = data这样的赋值，Python 中任何对象都能赋值给属性，列表也不例外
#这里不能这样写是因为engine.py里面要使用data做*这样的数值运算

ys = [1.0, -1.0, -1.0, 1.0]

# 1. 建网络
model = MLP(2,[4,4,44,4,1])#这里似乎可以写的很随意？最后一层我很迷惑，似乎只能写1，我想xs的张量突然就从2到ys的1
#这其中神经元内部存储数据维度的变化
#这里要看MLP的构造函数和call函数，我们容易发现sz[]列表中相邻的两个数，前者作为输入，控制神经元内张量维度，后者作为输出控制该层神经元的个数
# 这里的nin我没理解错的话，就是要与xs每个神经元的维度是一样的，所以只能写2✔，看Neuron函数的权重数就可以知道

# 2. 训练循环
for step in range(100):

    # 前向传播
    ypred = [model(x)for x in xs]#这里要特别注意，model(x)可以追溯到neuron(x)，x是一个神经元的输入，所以model(x)的意思应该是
    #一个神经元的输入经过多次感知机MLP会得到怎样的输出，要得到整个的输出就需要[model(x) for x in xs]
    #ypred的输出类型，model(x)->layer(x)->neuron(x),(layer等是相应类的实例化)，neuron(x)返回out是Value类型,out是一个神经元的输出
    #layer(x)也返回out，这里out返回一层多个神经元的输出，是一个列表[out1,out2,...]，类似于xs，
    #不同的是元素是Value类型，不是[2.0, 3.0]这样的列表
    #model(x)做的其实是一个传递过程，不断调用layer(x),最后倒数第二层作为输入，最后一层作为最后的输出。
    #输出类型和layer一样。所以说loss函数每一个元素的计算，最后输出将是Value类型，所以loss可以调用backward()
    # ypred = model(x for x in xs)

    # 计算loss
    loss = sum((sy-qy)**2 for sy,qy in zip(ypred,ys))
    #loss函数返回值的类型，做追溯。ys是列表类型，列表元素为浮点数。看ypred的输出
    # loss = sum((sy-qy)**2) for sy,qy in zip(ypred,ys)这样写为什么不对，sum要求的参数输入和计算逻辑是怎样的？
    # 为了更好的理解，需要知道zip函数？zip([a,b,c],[1,2,3]) → [(a,1),(b,1),(c,3)]
    # 清零梯度
    for p in model.parameters():
        # p.grad.data.zero_()不能这样写，这是pytorch的写法
        p.grad = 0
    # 反向传播
    # ypred.grad = [(sy - qy)*2 for sy,qy in zip(ypred,ys)]
    # ypred.backward()
    loss.backward()#为什么能调用.backward()我们需要知道loss函数的返回值类型，
    # 更新权重 权重更新方法和激活函数都可以换用更合适的
    for p in model.parameters():
        # p -= p.grad * 0.01
        p.data -= p.grad * 0.01
    print(f"step {step}, loss {loss.data:.4f}")
