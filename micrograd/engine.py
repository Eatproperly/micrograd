#定义计算逻辑，把训练神经网络中的数学表达用代码实现出来
class Value:
    """ stores a single scalar value and its gradient """
    #构造函数，Karpathy的理解是从输出层出发，看它是由谁计算得到的，输入层参与计算被当作children，保存上一层的输入元素。如果从前向传播理解，看做父节点也是可以的，
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        #这里backward既作为value对象的属性，但是python的奇妙就在于此，python中函数是普通对象，可以赋值给任意的变量或属性，value作为类也是对象也可以作为函数使用，如nn.py中的mlp(x)
        #这里的backward通过函数来赋值，而且这个函数无返回值，是操作函数，可以看到初始化为none
        self._backward = lambda: None
        self._prev = set(_children)#children是元组，set(children)得到集合,children为空，则为空集合
        self._op = _op # the op that produced this node, for graphviz / debugging / etc操作符，画图用的
    #魔法函数，每一个运算后返回value类型的out
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)#字面意思，看other是否是value实例如果不是将它实例化
        out = Value(self.data + other.data, (self, other), '+')#对输出赋值，data、children、op，prev不用担心，init构造函数会在value实例化时自动执行，用children对prev赋值
        #这里就可以看到属性作为函数的情况了，最后赋值给out
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"#assert是断言，断言other为int和float中的一种，断言错了报错后面字符串的内容
        out = Value(self.data**other, (self,), f'**{other}')#python支持**为幂次方，f''是输出字符串{other}是格式化？children这里，输出值out是由self一个value实例得来self**常数，所以只有一个

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')#和pow相同

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    #反向传播实现逻辑
    def backward(self):
        
        # topological order all of the children in the graph图中子节点进行拓扑排序
        topo = []
        visited = set()
        def build_topo(v):#用dfs的方法递归出拓扑序列，递归结果自然时输出值在最后
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):#反转拓扑序列，从输出值开始遍历每一个节点更新梯度，实现反向传播
            v._backward()
    #这里所有都是python内置魔法函数，嗯正面计算无法计算python会反过来试试
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)#这里在_neg_实现

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1#这里在_pow_中实现

    def __repr__(self):#str
        return f"Value(data={self.data}, grad={self.grad})"
