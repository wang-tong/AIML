# Step3 线性分类
## 一、多入单出的单层神经网路
### 1.线性二分类
**公式**

$$a(z) = \frac{1}{1 + e^{-z}}$$

- 导数

$$a^{'}(z) = a(z)(1 - a(z))$$

具体求导过程可以参考8.1节。

- 输入值域

$$(-\infty, \infty)$$

- 输出值域

$$(0,1)$$

- 函数图像
![](images/4.png)

### 2、线性二分类的神经网络实现

**输入层**

输入经度(x1)和纬度(x2)两个特征：

$$
x=\begin{pmatrix}
x_{1} & x_{2}
\end{pmatrix}
$$

**权重矩阵**

输入是2个特征，输出一个数，则W的尺寸就是2x1：

$$
w=\begin{pmatrix}
w_{1} \\ w_{2}
\end{pmatrix}
$$

B的尺寸是1x1，行数永远是1，列数永远和W一样。

$$
b=\begin{pmatrix}
b_{1}
\end{pmatrix}
$$

**输出层**

$$
z = x \cdot w + b
=\begin{pmatrix}
    x_1 & x_2
\end{pmatrix}
\begin{pmatrix}
    w_1 \\ w_2
\end{pmatrix}
$$
$$
=x_1 \cdot w_1 + x_2 \cdot w_2 + b \tag{1}
$$
$$a = Logistic(z) \tag{2}$$

 **损失函数**

二分类交叉熵函损失数：

$$
loss(w,b) = -[yln a+(1-y)ln(1-a)] \tag{3}
$$

**代码实现**
由于以前我们的神经网络只会做线性回归，现在多了一个做分类的技能，所以我们加一个枚举类型，可以让调用者通过指定参数来控制神经网络的功能。
```Python
class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3,
```
然后在超参类里把这个新参数加在初始化函数里：
```Python
class HyperParameters(object):
    def __init__(self, eta=0.1, max_epoch=1000, batch_size=5, eps=0.1, net_type=NetType.Fitting):
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps
        self.net_type = net_type
```
再增加一个Logistic分类函数：
```Python
class Logistic(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a
```
新建一个类便于管理：
```Python
class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type
    # end def

    def MSE(self, A, Y, count):
        ...

    # for binary classifier
    def CE2(self, A, Y, count):
        ...
class NeuralNet(object):
    def __init__(self, params, input_size, output_size):
        self.params = params
        self.W = np.zeros((input_size, output_size))
        self.B = np.zeros((1, output_size))

    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        if self.params.net_type == NetType.BinaryClassifier:
            A = Sigmoid().forward(Z)
            return A
        else:
            return Z
```
**主过程**：
```Python
if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    # net
    params = HyperParameters(eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    input = 2
    output = 1
    net = NeuralNet(params, input, output)
    net.train(reader, checkpoint=1)
    # inference
    x_predicate = np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a = net.inference(x_predicate)
    print("A=", a)    
```
**几何原理**

我们再观察一下下面这张分类正确的图：

![](images/5.png)

假设绿色方块为正类：标签值$y=1$，红色三角形为负类：标签值$y=0$。

从几何关系上理解，如果我们有一条直线，其公式为：$z = w \cdot x_1+b$，如图中的虚线所示，则所有正类的样本的x2都大于z，而所有的负类样本的x2都小于z，那么这条直线就是我们需要的分割线。用正例的样本来表示：

$$
x_2 > z，即正例满足条件：x_2 > w \cdot x_1 + b \tag{4}
$$

那么神经网络用矩阵运算+分类函数+损失函数这么复杂的流程，其工作原理是什么呢？

经典机器学习中的SVM确实就是用这种思路来解决这个问题的，即一个类别的所有样本在分割线的一侧，而负类样本都在线的另一侧。神经网络的正向公式如公式1，2所示，当a>0.5时，判为正类。当a<0.5时，判为负类。z=0即a=0.5时为分割线。
## 二、多入多出的单层神经网络
### 1、线性多分类问题

多分类问题一共有三种解法：
1. 一对一
   
每次先只保留两个类别的数据，训练一个分类器。如果一共有N个类别，则需要训练$C^2_N$个分类器。以N=3时举例，需要训练(A|B)，(B|C)，(A|C)三个分类器。

![](./images/one_vs_one.png)

如上图最左侧所示，这个二分类器只关心蓝色和绿色样本的分类，而不管红色样本的情况，也就是说在训练时，只把蓝色和绿色样本输入网络。
   
推理时，(A|B)分类器告诉你是A类时，需要到(A|C)分类器再试一下，如果也是A类，则就是A类。如果(A|C)告诉你是C类，则基本是C类了，不可能是B类，不信的话可以到(B|C)分类器再去测试一下。

2. 一对多
   
如下图，处理一个类别时，暂时把其它所有类别看作是一类，这样对于三分类问题，可以得到三个分类器。

![](./images/one_vs_multiple.png)

如最左图，这种情况是在训练时，把红色样本当作一类，把蓝色和绿色样本混在一起当作另外一类。

推理时，同时调用三个分类器，再把三种结果组合起来，就是真实的结果。比如，第一个分类器告诉你是“红类”，那么它确实就是红类；如果告诉你是非红类，则需要看第二个分类器的结果，绿类或者非绿类；依此类推。

3. 多对多

假设有4个类别ABCD，我们可以把AB算作一类，CD算作一类，训练一个分类器1；再把AC算作一类，BD算作一类，训练一个分类器2。
    
推理时，第1个分类器告诉你是AB类，第二个分类器告诉你是BD类，则做“与”操作，就是B类。
### 2、多分类函数- Softmax
### 2.1定义
Softmax加了个"soft"来模拟max的行为，但同时又保留了相对大小的信息。

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}}
$$
上式中:
- $z_j$是对第 j 项的分类原始值，即矩阵运算的结果
- $z_i$是参与分类计算的每个类别的原始值
- m 是总的分类数
- $a_j$是对第 j 项的计算结果
### 2.2python实现
**公式法**
```Python
def Softmax1(x):
    e_x = np.exp(x)
    v = np.exp(x) / np.sum(e_x)
    return v
```
**修改法**
```Python
class Softmax(object):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a

```
### 2.3线性多分类的神经网络实现  
**输入层**

输入经度(x1)和纬度(x2)两个特征：

$$
x=\begin{pmatrix}
x_1 & x_2
\end{pmatrix}
$$
**权重矩阵w/b**
W权重矩阵的尺寸，可以从后往前看，比如：输出层是3个神经元，输入层是2个特征，则W的尺寸就是3x2。

$$
w=\begin{pmatrix}
w_{11} & w_{12} & w_{13}\\
w_{21} & w_{22} & w_{23} 
\end{pmatrix}
$$
b的尺寸是1x3，列数永远和神经元的数量一样，行数永远是1。
$$
b=\begin{pmatrix}
b_1 & b_2 & b_3 
\end{pmatrix}
$$
**输出层**
输出层三个神经元，再加上一个Softmax计算，最后有A1,A2,A3三个输出，写作：

$$
z = \begin{pmatrix}z_1 & z_2 & z_3 \end{pmatrix}
$$
$$
a = \begin{pmatrix}a_1 & a_2 & a_3 \end{pmatrix}
$$

其中，$Z=X \cdot W+B，A = Softmax(Z)$
**代码实现**

```Python
if __name__ == '__main__':
    num_category = 3
    reader = SimpleDataReader()
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(num_category, base=1)

    num_input = 2
    params = HyperParameters(num_input, num_category, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet(params)
    net.train(reader, checkpoint=1)

    inference(net, reader)
```
## 三、总结
    这次我们学习了神经网络中的分类问题而它在很多资料中都称之为逻辑回归，Logistic Regression，其原因是使用了线性回归中的线性模型，加上一个Logistic二分类函数，共同构造了一个分类器。我们在本书中统称之为分类。本次学习中，我们从最简单的线性二分类开始学习，包括其原理，实现，训练过程，推理过程等等，并且以可视化的方式来帮助大家更好地理解这些过程。我们学习了实现逻辑非门，线性多分类的学习。多分类时，可以一对一、一对多、多对多，其中Softmax函数是多分类问题的分类函数，通过对它的分析，我们学习多分类的原理、实现、以及可视化结果，从而理解神经网络的工作方式。
