# Step3 - LinearClassification
#  线性二分类

## 逻辑回归模型

回归问题可以分为两类：线性回归和逻辑回归。在第二步中，我们学习了线性回归模型，在第三步中，我们将一起学习逻辑回归模型。

逻辑回归的英文是Logistic Regression，逻辑回归是用来计算“事件=Success”和“事件=Failure”的概率。当因变量的类型属于二元（1 / 0，真/假，是/否）变量时，我们就应该使用逻辑回归。

回忆线性回归，使用一条直线拟合样本数据，而逻辑回归是“拟合”0或1两个数值，而不是具体的连续数值，所以它叫广义线性模型。逻辑回归又称logistic回归分析，常用于数据挖掘，疾病自动诊断，经济预测等领域。

例如，探讨引发疾病的危险因素，并根据危险因素预测疾病发生的概率等。以胃癌病情分析为例，选择两组人群，一组是胃癌组，一组是非胃癌组，两组人群必定具有不同的体征与生活方式等。因此因变量就为是否胃癌，值为“是”或“否”；自变量就可以包括很多了，如年龄、性别、饮食习惯、幽门螺杆菌感染等。

自变量既可以是连续的，也可以是分类的。然后通过logistic回归分析，可以得到自变量的权重，从而可以大致了解到底哪些因素是胃癌的危险因素。同时根据该权值可以根据危险因素预测一个人患癌症的可能性。

逻辑回归的另外一个名字叫做分类器，分为线性分类器和非线性分类器，本章中我们学习线性分类器。而无论是线性还是非线性分类器，又分为两种：二分类问题和多分类问题，在本章中我们学习二分类问题。线性多分类问题将会在下一章讲述，非线性分类问题在后续的步骤中讲述。

综上所述，我们本章要学习的路径是：回归问题->逻辑回归问题->线性逻辑回归即分类问题->线性二分类问题。

## 二分类函数

此函数对线性和非线性二分类都适用。

## 二分类函数

对率函数Logistic Function，即可以做为激活函数使用，又可以当作二分类函数使用。而在很多不太正规的文字材料中，把这两个概念混用了，比如下面这个说法：“我们在最后使用Sigmoid激活函数来做二分类”，这是不恰当的。在本书中，我们会根据不同的任务区分激活函数和分类函数这两个概念，在二分类任务中，叫做Logistic函数，而在作为激活函数时，叫做Sigmoid函数。

- 公式

$$a(z) = \frac{1}{1 + e^{-z}}$$

- 导数

$$a^{'}(z) = a(z)(1 - a(z))$$

具体求导过程可以参考8.1节。

- 输入值域

$$(-\infty, \infty)$$

- 输出值域

$$(0,1)$$


- 使用方式

此函数实际上是一个概率计算，它把$(-\infty, \infty)$之间的任何数字都压缩到$(0,1)$之间，返回一个概率值，这个概率值接近1时，认为是正例，否则认为是负例。

训练时，一个样本x在经过神经网络的最后一层的矩阵运算结果作为输入z，经过Logistic计算后，输出一个$(0,1)$之间的预测值。我们假设这个样本的标签值为0属于负类，如果其预测值越接近0，就越接近标签值，那么误差越小，反向传播的力度就越小。

推理时，我们预先设定一个阈值比如0.5，则当推理结果大于0.5时，认为是正类；小于0.5时认为是负类；等于0.5时，根据情况自己定义。阈值也不一定就是0.5，也可以是0.65等等，阈值越大，准确率越高，召回率越低；阈值越小则相反，准确度越低，召回率越高。

比如：
- input=2时，output=0.88，而0.88>0.5，算作正例
- input=-1时，output=0.27，而0.27<0.5，算作负例

##  正向传播

### 矩阵运算

$$
z=x \cdot w + b \tag{1}
$$

### 分类计算

$$
a = Logistic(z)={1 \over 1 + e^{-z}} \tag{2}
$$

### 损失函数计算

二分类交叉熵损失函数：

$$
loss(w,b) = -[y \ln a+(1-y)\ln(1-a)] \tag{3}
$$

##  反向传播

### 求损失函数loss对a的偏导

$$
\frac{\partial loss}{\partial a}=-[{y \over a}+{-(1-y) \over 1-a}]=\frac{a-y}{a(1-a)} \tag{4}
$$

### 求损失函数a对z的偏导

$$
\frac{\partial a}{\partial z}= a(1-a) \tag{5}
$$

### 求损失函数loss对z的偏导

使用链式法则链接公式4和公式5：

$$
\frac{\partial loss}{\partial z}=\frac{\partial loss}{\partial a}\frac{\partial a}{\partial z}
$$
$$
=\frac{a-y}{a(1-a)} \cdot a(1-a)=a-y \tag{6}
$$

我们惊奇地发现，使用交叉熵函数求导得到的分母，与Logistic分类函数求导后的结果，正好可以抵消，最后只剩下了$a-y$这一项。真的有这么巧合的事吗？实际上这是依靠科学家们的聪明才智寻找出了这种匹配关系，以满足以下条件：
1. 损失函数满足二分类的要求，无论是正例还是反例，都是单调的；
2. 损失函数可导，以便于使用反向传播算法；
3. 让计算过程非常简单，一个减法就可以搞定。

### 多样本情况

我们用三个样本做实例化推导：

$$Z=
\begin{pmatrix}
  z_1 \\ z_2 \\ z_3
\end{pmatrix},
A=logistic\begin{pmatrix}
  z_1 \\ z_2 \\ z_3
\end{pmatrix}=
\begin{pmatrix}
  a_1 \\ a_2 \\ a_3
\end{pmatrix}
$$
$$
J(w,b)= -[y_1 \ln a_1+(1-y_1)\ln(1-a_1)] 
$$
$$
-[y_2 \ln a_2+(1-y_2)\ln(1-a_2)] 
$$
$$
-[y_3 \ln a_3+(1-y_3)\ln(1-a_3)] 
$$

$$
{\partial J(w,b) \over \partial Z}=
\begin{pmatrix}
  {\partial J(w,b) / \partial z_1} \\
  {\partial J(w,b) / \partial z_2} \\
  {\partial J(w,b) / \partial z_3}
\end{pmatrix} \tag{代入公式6结果}
$$
$$
=\begin{pmatrix}
  a_1-y_1 \\
  a_2-y_2 \\
  a_3-y_3 
\end{pmatrix}=A-Y
$$

所以，用矩阵运算时可以简化为矩阵相减的形式：$A-Y$。

## 对数几率的来历

经过数学推导后可以知道，神经网络实际也是在做这样一件事：经过调整w和b的值，把所有正例的样本都归纳到大于0.5的范围内，所有负例都小于0.5。但是如果只说大于或者小于，无法做准确的量化计算，所以用一个对率函数来模拟。

说到对率函数，还有一个问题，它为啥叫做“对数几率”函数呢？从哪里看出是“对数”了？“几率”是什么意思呢？

我们举例说明：假设有一个硬币，抛出落地后，得到正面的概率是0.5，得到反面的概率是0.5，这两个概率叫做probability。如果用正面的概率除以反面的概率，0.5/0.5=1，这个数值叫做odds，几率。

泛化一下，如果正面的概率是a，则反面的概率就是1-a，则几率等于：

$$odds = \frac{a}{1-a} \tag{9}$$

上式中，如果a是把样本x的预测为正例的可能性，那么1-a就是其负例的可能性，a/(1-a)就是正负例的比值，称为几率(odds)，反映了x作为正例的相对可能性，而对几率取对数就叫做对数几率(log odds, logit)。

如果假设概率如下表：

|a|0|0.1|0.2|0.3|0.4|0.5|0.6|0.7|0.8|0.9|1|
|--|--|--|--|--|--|--|--|--|--|--|--|
|1-a|1|0.9|0.8|0.7|0.6|0.5|0.4|0.3|0.2|0.1|0|
|odds|0|0.11|0.25|0.43|0.67|1|1.5|2.33|4|9|无穷大|
|ln(odds)|N/A|-2.19|-1.38|-0.84|-0.4|0|0.4|0.84|1.38|2.19|N/A|

可以看到0dds的值不是线性的，不利于分析问题，所以在表中第4行对odds取对数，可以得到一组成线性关系的值，即：

$$\ln \frac{a}{1-a} = xw + b \tag{10}$$

对公式10两边取自然指数：

$$\frac{a}{1-a}=e^{xw+b} \tag{11}$$

对公式11取倒数：

$$\frac{1-a}{a}=e^{-(xw+b)}$$

变形：

$$\frac{1}{a}-1=e^{-(xw+b)}$$
$$\frac{1}{a}=1+e^{-(xw+b)}$$
$$a=\frac{1}{1+e^{-(xw+b)}}$$

令$z=e^{-(xw+b)}$：

$$a=\frac{1}{1+e^{-z}} \tag{12}$$

公式12就是公式2！对数几率的函数形式可以认为是这样得到的。

以上推导过程，实际上就是用线性回归模型的预测结果来逼近样本分类的对数几率。这就是为什么它叫做逻辑回归(logistic regression)，但其实是分类学习的方法。这种方法的优点如下：

- 直接对分类可能性建模，无需事先假设数据分布，避免了假设分布不准确所带来的问题
- 不仅预测出类别，而是得到了近似的概率，这对许多需要利用概率辅助决策的任务很有用
- 对率函数是任意阶可导的凸函数，有很好的数学性，许多数值优化算法都可以直接用于求取最优解



### Python代码来实现实现推导过程
- 计算w值
  
```Python
def method1(X,Y,m):
    x_mean = X.mean()
    p = sum(Y*(X-x_mean))
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

def method2(X,Y,m):
    x_mean = X.mean()
    y_mean = Y.mean()
    p = sum(X*(Y-y_mean))
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

def method3(X,Y,m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w
```

- 计算b值

```Python
def calculate_b_1(X,Y,w,m):
    b = sum(Y-w*X)/m
    return b
def calculate_b_2(X,Y,w):
    b = Y.mean() - w * X.mean()
    return b
```
####（3）运算结果

```Python
if __name__ == '__main__':

    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    m = X.shape[0]
    w1 = method1(X,Y,m)
    b1 = calculate_b_1(X,Y,w1,m)

    w2 = method2(X,Y,m)
    b2 = calculate_b_2(X,Y,w2)

    w3 = method3(X,Y,m)
    b3 = calculate_b_1(X,Y,w3,m)

    print("w1=%f, b1=%f" % (w1,b1))
    print("w2=%f, b2=%f" % (w2,b2))
    print("w3=%f, b3=%f" % (w3,b3))
```
用以上几种方法，最后得出的结果都是一致的，可以起到交叉验证的作用：

```
w1=2.056827, b1=2.965434
w2=2.056827, b2=2.965434
w3=2.056827, b3=2.965434
```
###  梯度下降法
### （数学原理
在下面的公式中，我们规定x是样本特征值（单特征），y是样本标签值，z是预测值，下标 $i$ 表示其中一个样本。
### 预设函数（Hypothesis Function）
为一个线性函数：

$$z_i = x_i \cdot w + b \tag{1}$$

### 损失函数（Loss Function）
为均方差函数：
$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2 \tag{2}$$
与最小二乘法比较可以看到，梯度下降法和最小二乘法的模型及损失函数是相同的，都是一个线性模型加均方差损失函数，模型用于拟合，损失函数用于评估效果。
梯度计算
### 计算z的梯度
根据公式2：
$$
{\partial loss \over \partial z_i}=z_i - y_i \tag{3}
$$
### 计算w的梯度

我们用loss的值作为误差衡量标准，通过求w对它的影响，也就是loss对w的偏导数，来得到w的梯度。由于loss是通过公式2->公式1间接地联系到w的，所以我们使用链式求导法则，通过单个样本来求导。

根据公式1和公式3：

$$
{\partial{loss} \over \partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i \tag{4}
$$
### 计算b的梯度
$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i \tag{5}
$$
###  代码实现
```Python
if __name__ == '__main__':

    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    eta = 0.1
    w, b = 0.0, 0.0
    for i in range(reader.num_train):
        # get x and y value for one sample
        xi = X[i]
        yi = Y[i]
        zi = xi * w + b
        dz = zi - yi
        dw = dz * xi
        db = dz
        # update w,b
        w = w - eta * dw
        b = b - eta * db
    print("w=", w)    
    print("b=", b)
```
大家可以看到，在代码中，我们完全按照公式推导实现了代码，所以，大名鼎鼎的梯度下降，其实就是把推导的结果转化为数学公式和代码，直接放在迭代过程里！另外，我们并没有直接计算损失函数值，而只是把它融入在公式推导中。
###  运行结果
```
w= [1.71629006]
b= [3.19684087]
```
### 神经网络法
### 神经网络结构

我们是首次尝试建立神经网络，先用一个最简单的单层单点神经元：
![](./media/Setup.png)
### 输入层

此神经元在输入层只接受一个输入特征，经过参数w,b的计算后，直接输出结果。这样一个简单的“网络”，只能解决简单的一元线性回归问题，而且由于是线性的，我们不需要定义激活函数，这就大大简化了程序，而且便于大家循序渐进地理解各种知识点。

严格来说输入层在神经网络中并不能称为一个层。

### 权重w/b

因为是一元线性问题，所以w/b都是一个标量。

### 输出层

输出层1个神经元，线性预测公式是：

$$z_i = x_i \cdot w + b$$

z是模型的预测输出，y是实际的样本标签值，下标 $i$ 为样本。

### 损失函数

因为是线性回归问题，所以损失函数使用均方差函数。

$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2$$
### python程序实现
### 定义类
```Python
class NeuralNet(object):
    def __init__(self, eta):
        self.eta = eta
        self.w = 0
        self.b = 0
```
NeuralNet类从object类派生，并具有初始化函数，其参数是eta，也就是学习率，需要调用者指定。另外两个成员变量是w和b，初始化为0。
#### 前向计算

```Python
    def __forward(self, x):
        z = x * self.w + self.b
        return z
```
这是一个私有方法，所以前面有两个下划线，只在NeuralNet类中被调用，不对外公开。

### 反向传播

下面的代码是通过梯度下降法中的公式推导而得的，也设计成私有方法：

```Python
    def __backward(self, x,y,z):
        dz = z - y
        db = dz
        dw = x * dz
        return dw, db
```
dz是中间变量，避免重复计算。dz又可以写成delta_Z，是当前层神经网络的反向误差输入。
### 梯度更新

```Python
    def __update(self, dw, db):
        self.w = self.w - self.eta * dw
        self.b = self.b - self.eta * db
```
### 主程序
```Python
if __name__ == '__main__':
    # read data
    sdr = SimpleDataReader()
    sdr.ReadData()
    # create net
    eta = 0.1
    net = NeuralNet(eta)
    net.train(sdr)
    # result
    print("w=%f,b=%f" %(net.w, net.b))
    # predication
    result = net.inference(0.346)
    print("result=", result)
    ShowResult(net, sdr)
```
### 运行结果
打印输出结果：
```
w=1.716290,b=3.196841
result= [3.79067723]
```
###  多变量线性回归问题

# 多元线性回归模型
准则：
1. 自变量对因变量必须有显著的影响，并呈密切的线性相关；
2. 自变量与因变量之间的线性相关必须是真实的，而不是形式上的；
3. 自变量之间应具有一定的互斥性，即自变量之间的相关程度不应高于自变量与因变量之因的相关程度；
4. 自变量应具有完整的统计数据，其预测值容易确定。

|方法|正规方程|梯度下降|
|---|-----|-----|
|原理|几次矩阵运算|多次迭代|
|特殊要求|$X^TX$的逆矩阵存在|需要确定学习率|
|复杂度|$O(n^3)$|$O(n^2)$|
|适用样本数|$m \lt 10000$|$m \ge 10000$|

#  正规方程解法（Normal Equations)
####  推导方法

在做函数拟合（回归）时，我们假设函数H为：

$$h(w,b) = b + x_1 w_1+x_2 w_2+...+x_n w_n \tag{2}$$

令$b=w_0$，则：

$$h(w) = w_0 + x_1 \cdot w_1 + x_2 \cdot w_2+...+ x_n \cdot w_n\tag{3}$$

公式3中的x是一个样本的n个特征值，如果我们把m个样本一起计算，将会得到下面这个矩阵：

$$H(w) = X \cdot W \tag{4}$$

公式5中的X和W的矩阵形状如下：

$$
X^{(m \times (n+1))} = 
\begin{pmatrix} 
1 & x_{1,1} & x_{1,2} & \dots & x_{1,n} \\
1 & x_{2,1} & x_{2,2} & \dots & x_{2,n} \\
\dots \\
1 & x_{m,1} & x_{m,2} & \dots & x_{m,n}
\end{pmatrix} \tag{5}
$$

$$
W^{(n+1)}= \begin{pmatrix}
w_0 \\
w_1 \\
\dots \\
 w_n
\end{pmatrix}  \tag{6}
$$

然后我们期望假设函数的输出与真实值一致，则有：

$$H(w) = X \cdot W = Y \tag{7}$$

其中，Y的形状如下：

$$
Y^{(m)}= \begin{pmatrix}
y_1 \\
y_2 \\
\dots \\
y_m
\end{pmatrix}  \tag{8}
$$


直观上看，W = Y/X，但是这里三个值都是矩阵，而矩阵没有除法，所以需要得到X的逆矩阵，用Y乘以X的逆矩阵即可。但是又会遇到一个问题，只有方阵才有逆矩阵，而X不一定是方阵，所以要先把左侧变成方阵，就可能会有逆矩阵存在了。所以，先把等式两边同时乘以X的转置矩阵，以便得到X的方阵：

$$X^T X W = X^T Y \tag{9}$$

其中，$X^T$是X的转置矩阵，$X^T X$一定是个方阵，并且假设其存在逆矩阵，把它移到等式右侧来：

$$W = (X^T X)^{-1}{X^T Y} \tag{10}$$

至此可以求出W的正规方程。
#### 代码实现
```Python
if __name__ == '__main__':
    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    num_example = X.shape[0]
    one = np.ones((num_example,1))
    x = np.column_stack((one, (X[0:num_example,:])))
    a = np.dot(x.T, x)
    # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    #print(e)
    b=e[0,0]
    w1=e[1,0]
    w2=e[2,0]
    print("w1=", w1)
    print("w2=", w2)
    print("b=", b)
    # inference
    z = w1 * 15 + w2 * 93 + b
    print("z=",z)
```
#### 运行结果
```
w1= -2.0184092853092226
w2= 5.055333475112755
b= 46.235258613837644
z= 486.1051325196855
```
#### 神经网络解法
#### 定义神经网络结构

我们定义一个一层的神经网络，输入层为2或者更多，反正大于2了就没区别。这个一层的神经网络的特点是：
1. 没有中间层，只有输入项和输出层（输入项不算做一层），
2. 输出层只有一个神经元，
3. 神经元有一个线性输出，不经过激活函数处理，即在下图中，经过$\Sigma$求和得到Z值之后，直接把Z值输出。

与上一章的神经元相比，这次仅仅是多了一个输入，但却是质的变化，即，一个神经元可以同时接收多个输入，这是神经网络能够处理复杂逻辑的根本。

![](./media/setup.png)

#### 代码实现

公式6和第4.4节中的公式5一模一样，所以我们依然采用第四章中已经写好的HelperClass目录中的那些类，来表示我们的神经网络。虽然此次神经元多了一个输入，但是不用改代码就可以适应这种变化，因为在前向计算代码中，使用的是矩阵乘的方式，可以自动适应x的多个列的输入，只要对应的w的矩阵形状是正确的即可。

但是在初始化时，我们必须手动指定x和w的形状，如下面的代码所示：

```Python
from HelperClass.SimpleDataReader import *

if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    # net
    params = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=1, eps = 1e-5)
    net = NeuralNet(params)
    net.train(reader)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    print(net.inference(x))
```
在参数中，指定了学习率0.1，最大循环次数100轮，批大小1个样本，以及停止条件损失函数值1e-5。

在神经网络初始化时，指定了input_size=2，且output_size=1，即一个神经元可以接收两个输入，最后是一个输出。

最后的inference部分，是把两个条件（15公里，93平方米）代入，查看输出结果。

在下面的神经网络的初始化代码中，W的初始化是根据input_size和output_size的值进行的。

```Python
class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.W = np.zeros((self.params.input_size, self.params.output_size))
        self.B = np.zeros((1, self.params.output_size))
```

#### 正向计算的代码
```Python
class NeuralNet(object):
    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        return Z
```

#### 误差反向传播的代码

```Python
class NeuralNet(object):
    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB
```

####  运行结果

在Visual Studio 2017中，可以使用Ctrl+F5运行Level2的代码，但是，会遇到一个令人沮丧的打印输出：

```
epoch=0
NeuralNet.py:32: RuntimeWarning: invalid value encountered in subtract
  self.W = self.W - self.params.eta * dW
0 500 nan
epoch=1
1 500 nan
epoch=2
2 500 nan
epoch=3
3 500 nan
......
```
#### 样本特征数据归一化
#### 基本概念

有三个类似的概念，归一化，标准化，中心化。

#### 归一化

把数据线性地变成[0,1]或[-1,1]之间的小数，把带单位的数据（比如米，公斤）变成无量纲的数据，区间缩放。

归一化有三种方法:

1. Min-Max归一化：
$$x_{new}={x-x_{min} \over x_{max} - x_{min}} \tag{1}$$

2. 平均值归一化
   
$$x_{new} = {x - \bar{x} \over x_{max} - x_{min}} \tag{2}$$

3. 非线性归一化

对数转换：
$$y=log(x) \tag{3}$$

反余切转换：
$$y=atan(x) \cdot 2/π  \tag{4}$$

#### 标准化

把每个特征值中的所有数据，变成平均值为0，标准差为1的数据，最后为正态分布。Z-score规范化（标准差标准化 / 零均值标准化，其中std是标准差）：

$$x_{new} = (x - \bar{x})／std \tag{5}$$

#### 中心化

平均值为0，无标准差要求：
$$x_{new} = x - \bar{x} \tag{6}$$
#### 代码实现

在HelperClass目录的SimpleDataReader.py文件中，给该类增加一个方法：

```Python
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((2,num_feature))
        # 按列归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            # get one feature from all examples
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            # min value
            self.X_norm[0,i] = min_value 
            # range value
            self.X_norm[1,i] = max_value - min_value 
            new_col = (col_i - self.X_norm[0,i])/(self.X_norm[1,i])
            X_new[:,i] = new_col
        #end for
        self.XTrain = X_new
```
#### 运行结果
运行上述代码，看打印结果：
```
epoch=9
9 0 391.75978721600353
9 100 387.79811202735783
9 200 502.9576560855685
9 300 395.63883403610765
9 400 417.61092908059885
9 500 404.62859838907883
9 600 398.0285538622818
9 700 469.12489440138637
9 800 380.78054509441193
9 900 575.5617634691969
W= [[-41.71417524]
 [395.84701164]]
B= [[242.15205099]]
z= [[37366.53336103]]
```

# 总结
本次课学习的是很多神经网络的常用方法，我在学习过程中较为吃力，而且在本码运行出现了很大的问题，最后得以在同学的帮助下才完成，我深知自己能力的不足，从今往后必会紧紧跟随老师的脚步，学习上的漏洞定会及时找同学们请教，尽量缩短与优秀同学的差距。