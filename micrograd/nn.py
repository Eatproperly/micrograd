import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):#神经元
     # 构造函数 nin是每个神经元张量维数，可以理解为一个列向量组，nonlin是非线性
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]#这里的value调用engine.py，那里定义好了所有计算规则。使用*、+等都会自动调用相应魔法函数
        self.b = Value(0)
        self.nonlin = nonlin
    #魔法函数 Neuron的对象可以当函数调用输入参数x，x是列表，元素个数与nin相同，通过relu之类的函数将值限制在一定范围内得到激活值返回
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    #返回参数列表 w是列表[]将b变成列表拼接
    def parameters(self):
        return self.w + [self.b]
    #调试用的 非线性输出RELU线性输出Linear
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):#神经元某一层
    #nin还是神经元张量维数，nout是输出刚好表示该层神经元数目，kwargs={'nonlin': False}，**kwargs把字典解包成参数？？？
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]#生成nout数的Neuron,搞成一个列表？[]?
    #神经元的更上一级,每个Neuron调用它的call魔法函数，把激活值也就是这一层的输出值拼成一个列表
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    #修改 依旧嵌套neuron，输出layer的参数列表
    def parameters(self):
        result = []
        for n in self.neurons:
            for p in n.parameters():
                result.append(p)
        return result
    #为了调试，其中str(neuron)魔法函数会默认调用__repr__
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    #nin与上面两个相同，表示输入数，也即神经元内的张量维数或者说向量的元素个数，nouts是[]表示每层的神经元个数拼成一个列表
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]#通过nonlin让最后一层不做relu处理，因为要输出结果开放式而不是二元分布式
    #MLP的call函数依旧嵌套，多了一层嵌套而已，调用layer()，输出只输出最后一层的结果，发现它实现了前向传播的效果，得到了最后的输出值，你只需要调用MLP的对象mlp，mlp([])，[]中元素数与nin相同，作为输入，
    #可以这样理解：它是不存在的一层，这一层没有输入，不进行计算，手动输入值就是它的输出，从而作为下一层的输入
    #训练过程中我只要调用mlp(x),x=[,,],即可完成前向传播，得到最后一层输出值
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    #得到所有参数，反向传播更新参数时使用
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    #调试用的，给自己看 
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
