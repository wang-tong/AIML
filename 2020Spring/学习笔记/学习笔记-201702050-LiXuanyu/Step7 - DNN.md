# Step7 - DNN
#  搭建深度神经网络框架
## 功能/模式分析
比较第12章中的三层神经网络的代码，我们可以看到大量的重复之处，比如前向计算中：

```Python
def forward3(X, dict_Param):
    ...
    # layer 1
    Z1 = np.dot(W1,X) + B1
    A1 = Sigmoid(Z1)
    # layer 2
    Z2 = np.dot(W2,A1) + B2
    A2 = Tanh(Z2)
    # layer 3
    Z3 = np.dot(W3,A2) + B3
    A3 = Softmax(Z3)
    ...    
```

1，2，3三层的模式完全一样：矩阵运算+激活/分类函数。

再看看反向传播：

```Python
def backward3(dict_Param,cache,X,Y):
    ...
    # layer 3
    dZ3= A3 - Y
    dW3 = np.dot(dZ3, A2.T)
    dB3 = np.sum(dZ3, axis=1, keepdims=True)
    # layer 2
    dZ2 = np.dot(W3.T, dZ3) * (1-A2*A2) # tanh
    dW2 = np.dot(dZ2, A1.T)
    dB2 = np.sum(dZ2, axis=1, keepdims=True)
    # layer 1
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1-A1)   #sigmoid
    dW1 = np.dot(dZ1, X.T)
    dB1 = np.sum(dZ1, axis=1, keepdims=True)
    ...
```
每一层的模式也非常相近：计算本层的dZ，再根据dZ计算dW和dB。

因为三层网络比两层网络多了一层，所以会在初始化、前向、反向、更新参数等四个环节有所不同，但却是有规律的。再加上前面章节中，为了实现一些辅助功能，我们已经写了很多类。所以，现在可以动手搭建一个深度学习的mini框架了。

#### NeuralNet

我们首先需要一个NeuralNet类，来包装基本的神经网络结构和功能：

- Layers - 神经网络各层的容器，按添加顺序维护一个列表
- Parameters - 基本参数，包括普通参数和超参
- Loss Function - 提供计算损失函数值，存储历史记录并最后绘图的功能
- LayerManagement() - 添加神经网络层
- ForwardCalculation() - 调用各层的前向计算方法
- BackPropagation() - 调用各层的反向传播方法
- PreUpdateWeights() - 预更新各层的权重参数
- UpdateWeights() - 更新各层的权重参数
- Train() - 训练
- SaveWeights() - 保存各层的权重参数
- LoadWeights() - 加载各层的权重参数

#### Layer

是一个抽象类，以及更加需要增加的实际类，包括：
- Fully Connected Layer
- Classification Layer
- Activator Layer
- Dropout Layer
- Batch Norm Layer

将来还会包括：
- Convolution Layer
- Max Pool Layer

每个Layer都包括以下基本方法：
 - ForwardCalculation() - 调用本层的前向计算方法
 - BackPropagation() - 调用本层的反向传播方法
 - PreUpdateWeights() - 预更新本层的权重参数
 - UpdateWeights() - 更新本层的权重参数
 - SaveWeights() - 保存本层的权重参数
 - LoadWeights() - 加载本层的权重参数

#### Activator Layer

激活函数和分类函数：

- Identity - 直传函数，即没有激活处理
- Sigmoid
- Tanh
- Relu

#### Classification Layer

分类函数，包括：
- Sigmoid二分类
- Softmax多分类


 #### Parameters

 基本神经网络运行参数：

 - 学习率
 - 最大epoch
 - batch size
 - 损失函数定义
 - 初始化方法
 - 优化器类型
 - 停止条件
 - 正则类型和条件

#### LossFunction

损失函数及帮助方法：

- 均方差函数
- 交叉熵函数二分类
- 交叉熵函数多分类
- 记录损失函数
- 显示损失函数历史记录
- 获得最小函数值时的权重参数

#### Optimizer

优化器：

- SGD
- Momentum
- Nag
- AdaGrad
- AdaDelta
- RMSProp
- Adam

#### WeightsBias

权重矩阵，仅供全连接层使用：

- 初始化 
  - Zero, Normal, MSRA (HE), Xavier
  - 保存初始化值
  - 加载初始化值
- Pre_Update - 预更新
- Update - 更新
- Save - 保存训练结果值
- Load - 加载训练结果值

#### DataReader

样本数据读取器：

- ReadData - 从文件中读取数据
- NormalizeX - 归一化样本值
- NormalizeY - 归一化标签值
- GetBatchSamples - 获得批数据
- ToOneHot - 标签值变成OneHot编码用于多分类
- ToZeorOne - 标签值变成0/1编码用于二分类
- Shuffle - 打乱样本顺序

从中派生出两个数据读取器：
- MnistImageDataReader - 读取MNIST数据
- CifarImageReader - 读取Cifar10数据


## 回归试验 - 万能近似定理

在第九章中，我们用一个两层的神经网络，验证了万能近似定理。当时是用hard code方式写的，现在我们用mini框架来搭建一下。

### 搭建模型

这个模型很简单，一个双层的神经网络，第一层后面接一个Sigmoid激活函数，第二层直接输出拟合数据：

<img src="./media/7/ch09_net.png" />

```Python
def model():
    dataReader = LoadData()
    num_input = 1
    num_hidden1 = 4
    num_output = 1

    max_epoch = 10000
    batch_size = 10
    learning_rate = 0.5

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.001))

    net = NeuralNet_4_0(params, "Level1_CurveFittingNet")
    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivationLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    fc2 = FcLayer_1_0(num_hidden1, num_output, params)
    net.add_layer(fc2, "fc2")

    net.train(dataReader, checkpoint=100, need_test=True)

    net.ShowLossHistory()
    ShowResult(net, dataReader)
```

超参数说明：

1. 输入层1个神经元，因为只有一个x值
2. 隐层4个神经元，对于此问题来说应该是足够了，因为特征很少
3. 输出层1个神经元，因为是拟合任务
4. 学习率=0.5
5. 最大epoch=10000轮
6. 批量样本数=10
7. 拟合网络类型
8. Xavier初始化
9. 绝对损失停止条件=0.001




##  神经网络反向传播四大公式

著名的反向传播四大公式是：

  $$\delta^{L} = \nabla_{a}C \odot \sigma_{'}(Z^L) \tag{80}$$
  $$\delta^{l} = ((W^{l + 1})^T\delta^{l+1})\odot\sigma_{'}(Z^l) \tag{81}$$
  $$\frac{\partial{C}}{\partial{b_j^l}} = \delta_j^l \tag{82}$$
  $$\frac{\partial{C}}{\partial{w_{jk}^{l}}} = a_k^{l-1}\delta_j^l \tag{83}$$

###  直观理解反向传播四大公式

下面我们用一个简单的两个神经元的全连接神经网络来直观解释一下这四个公式，

<img src="./media/7/bp.png" />

每个结点的输入输出标记如图上所示，使用MSE作为计算loss的函数，那么可以得到这张计算图中的计算过公式如下所示：

$$e_{01} = \frac{1}{2}(y-a_1^3)^2$$
$$a_1^3 = sigmoid(z_1^3)$$
$$z_1^3 = (w_{11}^2 \cdot a_1^2 + w_{12}^2 \cdot a_2^2 + b_1^3)$$
$$a_1^2 = sigmoid(z_1^2)$$
$$z_1^2 = (w_{11}^1 \cdot a_1^1 + w_{12}^1 \cdot a_2^1 + b_1^2)$$

我们按照反向传播中梯度下降的原理来对损失求梯度，计算过程如下：

$$\frac{\partial{e_{o1}}}{\partial{w_{11}^2}} = \frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}\frac{\partial{z_{1}^3}}{\partial{w_{11}^2}}=\frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}a_{1}^2$$

$$\frac{\partial{e_{o1}}}{\partial{w_{12}^2}} = \frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}\frac{\partial{z_{1}^3}}{\partial{w_{12}^2}}=\frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}a_{2}^2$$

$$\frac{\partial{e_{o1}}}{\partial{w_{11}^1}} = \frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}\frac{\partial{z_{1}^3}}{\partial{a_{1}^2}}\frac{\partial{a_{1}^2}}{\partial{z_{1}^2}}\frac{\partial{z_{1}^2}}{\partial{w_{11}^1}} = \frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}\frac{\partial{z_{1}^3}}{\partial{a_{1}^2}}\frac{\partial{a_{1}^2}}{\partial{z_{1}^2}}a_1^1$$

$$=\frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}w_{11}^2\frac{\partial{a_{1}^2}}{\partial{z_{1}^2}}a_1^1$$

$$\frac{\partial{e_{o1}}}{\partial{w_{12}^1}} = \frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}\frac{\partial{z_{1}^3}}{\partial{a_{2}^2}}\frac{\partial{a_{2}^2}}{\partial{z_{1}^2}}\frac{\partial{z_{1}^2}}{\partial{w_{12}^1}} = \frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}\frac{\partial{z_{1}^3}}{\partial{a_{2}^2}}\frac{\partial{a_{2}^2}}{\partial{z_{1}^2}}a_2^2$$

$$=\frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}w_{12}^2\frac{\partial{a_{2}^2}}{\partial{z_{1}^2}}a_2^2$$

上述式中，$\frac{\partial{a}}{\partial{z}}$是激活函数的导数，即$\sigma^{'}(z)$项。观察到在求偏导数过程中有共同项$\frac{\partial{e_{o1}}}{\partial{a_{1}^3}}\frac{\partial{a_{1}^3}}{\partial{z_{1}^3}}$,采用$\delta$符号记录,用矩阵形式表示，
即：

$$\delta^L = [\frac{\partial{e_{o1}}}{\partial{a_{i}^L}}\frac{\partial{a_{i}^L}}{\partial{z_{i}^L}}] = \nabla_{a}C\odot\sigma^{'}(Z^L)$$

上述式中，$[a_i]$表示一个元素是a的矩阵，$\nabla_{a}C$表示将损失$C$对$a$求梯度，$\odot$表示矩阵element wise的乘积（也就是矩阵对应位置的元素相乘）。

从上面的推导过程中，我们可以得出$\delta$矩阵的递推公式：

$$\delta^{L-1} = (W^L)^T[\frac{\partial{e_{o1}}}{\partial{a_{i}^L}}\frac{\partial{a_{i}^L}}{\partial{z_{i}^L}}]\odot\sigma^{'}(Z^{L - 1})$$

所以在反向传播过程中只需要逐层利用上一层的$\delta^l$进行递推即可。

相对而言，这是一个非常直观的结果，这份推导过程也是不严谨的。下面，我们会从比较严格的数学定义角度进行推导，首先要补充一些定义。


### 神经网络有关公式证明

+ 首先，来看一个通用情况，已知$f = A^TXB$，$A,B$是常矢量，希望得到$\frac{\partial{f}}{\partial{X}}$，推导过程如下

  根据式(94)，

  $$
  df = d(A^TXB) = d(A^TX)B + A^TX(dB) = d(A^TX)B + 0 = d(A^T)XB+A^TdXB = A^TdXB
  $$

  由于$df$是一个标量，标量的迹等于本身，同时利用公式(99):

  $$
  df = tr(df) = tr(A^TdXB) = tr(BA^TdX)
  $$

  由于公式(92):

  $$
  tr(df) = tr({(\frac{\partial{f}}{\partial{X}})}^TdX)
  $$

  可以得到:

  $$
  (\frac{\partial{f}}{\partial{X}})^T = BA^T
  $$
  $$
  \frac{\partial{f}}{\partial{X}} = AB^T \tag{101}
  $$

+ 我们来看全连接层的情况：

  $$ Y = WX + B$$

  取全连接层其中一个元素

  $$ y = wX + b$$

  这里的$w$是权重矩阵的一行，尺寸是$1 \times M$，X是一个大小为$M \times 1$的矢量，y是一个标量，若添加一个大小是1的单位阵，上式整体保持不变：

  $$ y = (w^T)^TXI + b$$

  利用式(92)，可以得到

  $$ \frac{\partial{y}}{\partial{X}} = I^Tw^T = w^T$$

  因此在误差传递的四大公式中，在根据上层传递回来的误差$\delta$继续传递的过程中，利用链式法则，有

  $$\delta^{L-1} = (W^L)^T \delta^L \odot \sigma^{'}(Z^{L - 1})$$

  同理，若将$y=wX+b$视作：

  $$ y = IwX + b $$

  那么利用式(92),可以得到：

  $$ \frac{\partial{y}}{\partial{w}} = X^T$$

+ 使用softmax和交叉熵来计算损失的情况下：

  $$ l = - Y^Tlog(softmax(Z))$$

  式中，$y$是数据的标签，$Z$是网络预测的输出，$y$和$Z$的维度是$N \times 1$。经过softmax处理作为概率。希望能够得到$\frac{\partial{l}}{\partial{Z}}$，下面是推导的过程：

  $$
  softmax(Z) = \frac{exp(Z)}{\boldsymbol{1}^Texp(Z)}
  $$

  其中， $\boldsymbol{1}$是一个维度是$N \times 1$的全1向量。将softmax表达式代入损失函数中，有

  $$
  dl = -Y^T d(log(softmax(Z)))\\
  = -Y^T d (log\frac{exp(Z)}{\boldsymbol{1}^Texp(Z)}) \\
  = -Y^T dZ + Y^T \boldsymbol{1}d(log(\boldsymbol{1}^Texp(Z))) \tag{102}
  $$

  下面来化简式(102)的后半部分,利用式(98)：

  $$
  d(log(\boldsymbol{1}^Texp(Z))) = log^{'}(\boldsymbol{1}^Texp(Z)) \odot dZ
  = \frac{\boldsymbol{1}^T(exp(Z)\odot dZ)}{\boldsymbol{1}^Texp(Z)}
  $$

  利用式(100)，可以得到

  $$
  tr(Y^T \boldsymbol{1}\frac{\boldsymbol{1}^T(exp(Z)\odot dZ)}{\boldsymbol{1}^Texp(Z)}) =
  tr(Y^T \boldsymbol{1}\frac{(\boldsymbol{1} \odot (exp(Z))^T dZ)}{\boldsymbol{1}^Texp(Z)})
  $$
  $$ =
  tr(Y^T \boldsymbol{1}\frac{exp(Z)^T dZ}{\boldsymbol{1}^Texp(Z)}) = tr(Y^T \boldsymbol{1} softmax(Z)^TdZ) \tag{103}
  $$

  将式(103)代入式(102)并两边取迹，可以得到：

  $$
  dl = tr(dl) = tr(-y^T dZ + y^T\boldsymbol{1}softmax(Z)^TdZ) = tr((\frac{\partial{l}}{\partial{Z}})^TdZ)
  $$

  在分类问题中，一个标签中只有一项会是1，所以$Y^T\boldsymbol{1} = 1$，因此有

  $$
  \frac{\partial{l}}{\partial{Z}} = softmax(Z) - Y
  $$

  这也就是在损失函数中计算反向传播的误差的公式。



#  权重矩阵初始化

权重矩阵初始化是一个非常重要的环节，是训练神经网络的第一步，选择正确的初始化方法会带了事半功倍的效果。这就好比攀登喜马拉雅山，如果选择从南坡登山，会比从北坡容易很多。而初始化权重矩阵，相当于下山时选择不同的道路，在选择之前并不知道这条路的难易程度，只是知道它可以抵达山下。这种选择是随机的，即使你使用了正确的初始化算法，每次重新初始化时也会给训练结果带来很多影响。

比如第一次初始化时得到权重值为(0.12847，0.36453)，而第二次初始化得到(0.23334，0.24352)，经过试验，第一次初始化用了3000次迭代达到精度为96%的模型，第二次初始化只用了2000次迭代就达到了相同精度。这种情况在实践中是常见的。

###  零初始化

即把所有层的W值的初始值都设置为0。

$$
W = 0
$$

但是对于多层网络来说，绝对不能用零初始化，否则权重值不能学习到合理的结果。看下面的零值初始化的权重矩阵值打印输出：
```
W= [[-0.82452497 -0.82452497 -0.82452497]]
B= [[-0.01143752 -0.01143752 -0.01143752]]
W= [[-0.68583865]
 [-0.68583865]
 [-0.68583865]]
B= [[0.68359678]]
```

可以看到W1、B1、W2内部3个单元的值都一样，这是因为初始值都是0，所以梯度均匀回传，导致所有w的值都同步更新，没有差别。这样的话，无论多少论，最终的结果也不会正确。

### 随机初始化

把W初始化均值为0，方差为1的矩阵：

$$
W \sim G \begin{bmatrix} 0, 1 \end{bmatrix}
$$

当目标问题较为简单时，网络深度不大，所以用随机初始化就可以了。但是当使用深度网络时，会遇到这样的问题：

<img src="./media/7/init_normal_sigmoid.png" ch="500" />

上图是一个6层的深度网络，使用全连接层+Sigmoid激活函数，图中表示的是各层激活函数的直方图。可以看到各层的激活值严重向两侧[0,1]靠近，从Sigmoid的函数曲线可以知道这些值的导数趋近于0，反向传播时的梯度逐步消失。处于中间地段的值比较少，对参数学习非常不利。

基于这种观察，Xavier Glorot等人研究出了下面的Xavier初始化方法。

### Xavier初始化方法

条件：正向传播时，激活值的方差保持不变；反向传播时，关于状态值的梯度的方差保持不变。

$$
W \sim U \begin{bmatrix} -\sqrt{{6 \over n_{input} + n_{output}}}, \sqrt{{6 \over n_{input} + n_{output}}} \end{bmatrix}
$$

假设激活函数关于0对称，且主要针对于全连接神经网络。适用于tanh和softsign。

即权重矩阵参数应该满足在该区间内的均匀分布。其中的W是权重矩阵，U是Uniform分布，即均匀分布。

### 小结

|ID|网络深度|初始化方法|激活函数|说明|
|---|---|---|---|---|
|1|单层|Zero|无|可以|
|2|双层层|Zero|Sigmoid|错误，不能进行正确的反向传播|
|3|双层|Normal|Sigmoid|可以|
|4|多层|Normal|Sigmoid|激活值分布成凹形，不利于反向传播|
|5|多层|Xavier|Sigmoid|正确|
|6|多层|Xavier|Relu|激活值分布偏向0，不利于反向传播|
|7|多层|MSRA|Relu|正确|

从上表可以看到，由于网络深度和激活函数的变化，使得人们不断地研究新的初始化方法来适应，最终得到1、3、5、7这几种组合。


# 批量归一化的原理

###  基本数学知识

#### 正态分布

正态分布，又叫做高斯分布。

若随机变量X，服从一个位置参数为μ、尺度参数为σ的概率分布，且其概率密度函数为：

$$
f(x)={1 \over \sigma\sqrt{2 \pi} } e^{- {(x-\mu)^2} \over 2\sigma^2} \tag{1}
$$

则这个随机变量就称为正态随机变量，正态随机变量服从的分布就称为正态分布，记作：

$$
X \sim N(\mu,\sigma^2) \tag{2}
$$

当μ=0,σ=1时，称为标准正态分布：

$$X \sim N(0,1) \tag{3}$$

此时公式简化为：

$$
f(x)={1 \over \sqrt{2 \pi}} e^{- {x^2} \over 2} \tag{4}
$$



### 批量归一化

既然可以把原始训练样本做归一化，那么如果在深度神经网络的每一层，都可以有类似的手段，也就是说把层之间传递的数据移到0点附近，那么训练效果就应该会很理想。这就是批归一化BN的想法的来源。

深度神经网络随着网络深度加深，训练起来越困难，收敛越来越慢，这是个在DL领域很接近本质的问题。很多论文都是解决这个问题的，比如ReLU激活函数，再比如Residual Network。BN本质上也是解释并从某个不同的角度来解决这个问题的。

BN就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同的分布，致力于将每一层的输入数据正则化成$N(0,1)$的分布。因次，每次训练的数据必须是mini-batch形式，一般取32，64等数值。

具体的数据处理过程如下图所示：

<img src="./media/7/bn6.png" ch="500" />

1. 数据在训练过程中，在网络的某一层会发生Internal Covariate Shift，导致数据处于激活函数的饱和区；
2. 经过均值为0、方差为1的变换后，位移到了0点附近。但是只做到这一步的话，会带来两个问题：
   
   a. 在[-1,1]这个区域，Sigmoid激活函数是近似线性的，造成激活函数失去非线性的作用；
   
   b. 在二分类问题中我们学习过，神经网络把正类样本点推向了右侧，把负类样本点推向了左侧，如果再把它们强行向中间集中的话，那么前面学习到的成果就会被破坏；

3. 经过$\gamma、\beta$的线性变换后，把数据区域拉宽，则激活函数的输出既有线性的部分，也有非线性的部分，这就解决了问题a；而且由于$\gamma、\beta$也是通过网络进行学习的，所以以前学到的成果也会保持，这就解决了问题b。

在实际的工程中，我们把BN当作一个层来看待，一般架设在全连接层（或卷积层）与激活函数层之间。

### 前向计算

#### 符号表

下表中，m表示batch_size的大小，比如32或64个样本/批；n表示features数量，即样本特征值数量。

|符号|数据类型|数据形状|
|:---------:|:-----------:|:---------:|
|$X$| 输入数据矩阵 | [m, n] |
|$x_i$|输入数据第i个样本| [1, n] |
|$N$| 经过归一化的数据矩阵 | [m, n] |
|$n_i$| 经过归一化的单样本 | [1, n] |
|$\mu_B$| 批数据均值 | [1, n] |
|$\sigma^2_B$| 批数据方差 | [1, n] |
|$m$|批样本数量| [1] |
|$\gamma$|线性变换参数| [1, n] |
|$\beta$|线性变换参数| [1, n] |
|$Z$|线性变换后的矩阵| [1, n] |
|$z_i$|线性变换后的单样本| [1, n] |
|$\delta$| 反向传入的误差 | [m, n] |

如无特殊说明，以下乘法为元素乘，即element wise的乘法。

在训练过程中，针对每一个batch数据，m是批的大小。进行的操作是，将这组数据正则化，之后对其进行线性变换。

具体的算法步骤是：

$$
\mu_B = {1 \over m}\sum_1^m x_i \tag{6}
$$

$$
\sigma^2_B = {1 \over m} \sum_1^m (x_i-\mu_B)^2 \tag{7}
$$

$$
n_i = {x_i-\mu_B \over \sqrt{\sigma^2_B + \epsilon}} \tag{8}
$$

$$
z_i = \gamma n_i + \beta \tag{9}
$$

其中，$\gamma 和 \beta$是训练出来的，$\epsilon$是防止$\mu_B^2$为0时加的一个很小的数值，通常为1e-5。


###  测试和推理时的归一化方法

批量归一化的“批量”两个字，表示在训练过程中需要有一小批数据，比如32个样本。而在测试过程或推理时，我们只有一个样本的数据，根本没有mini-batch的概念，无法计算算出正确的均值。因此，我们使用的均值和方差数据是在训练过程中样本值的平均。也就是：

$$
E[x] = E[\mu_B]
$$
$$
Var[x] = {m \over m-1} E[\sigma^2_B]
$$

一种做法是，我们把所有批次的$\mu$和$\sigma$都记录下来，然后在最后训练完毕时（或做测试时）平均一下。

另外一种做法是使用类似动量的方式，训练时，加权平均每个批次的值，权值$\alpha$可以为0.9：

$$m_{t} = \alpha \cdot m_{t-1} + (1-\alpha) \cdot \mu_t$$
$$v_{t} = \alpha \cdot v_{t-1} + (1-\alpha) \cdot \sigma_t$$

测试或推理时，直接使用$m_t和v_t$的值即可。

###  批量归一化的优点

1. 可以选择比较大的初始学习率，让你的训练速度提高。
   
    以前还需要慢慢调整学习率，甚至在网络训练到一定程度时，还需要想着学习率进一步调小的比例选择多少比较合适，现在我们可以采用初始很大的学习率，因为这个算法收敛很快。当然这个算法即使你选择了较小的学习率，也比以前的收敛速度快，因为它具有快速训练收敛的特性；

2. 减少对初始化的依赖
   
    一个不太幸运的初始化，可能会造成网络训练实际很长，甚至不收敛。

3. 减少对正则的依赖
   
   在第16章中，我们将会学习正则化知识，以增强网络的泛化能力。采用BN算法后，我们会逐步减少对正则的依赖，比如令人头疼的dropout、L2正则项参数的选择问题，或者可以选择更小的L2正则约束参数了，因为BN具有提高网络泛化能力的特性；



# 正则化

正则化的英文为Regularization，用于防止过拟合。

## 过拟合

###  拟合程度比较

在深度神经网络中，我们遇到的另外一个挑战，就是网络的泛化问题。所谓泛化，就是模型在测试集上的表现要和训练集上一样好。经常有这样的例子：一个模型在训练集上千锤百炼，能到达99%的准确率，拿到测试集上一试，准确率还不到90%。这说明模型过度拟合了训练数据，而不能反映真实世界的情况。解决过度拟合的手段和过程，就叫做泛化。

神经网络的两大功能：回归和分类。这两类任务，都会出现欠拟合和过拟合现象，如下图所示：

<img src="./media/7/fitting.png" />

上图是回归任务中的三种情况，依次为：欠拟合、正确的拟合、过拟合。

<img src="./media/7/classification.png" />

上图是分类任务中的三种情况，依次为：分类欠妥、正确的分类、分类过度。由于分类可以看作是对分类边界的拟合，所以我们经常也统称其为拟合。

上图中对于“深入敌后”的那颗绿色点样本，正确的做法是把它当作噪音看待，而不要让它对网络产生影响。而对于上例中的欠拟合情况，如果简单的（线性）模型不能很好地完成任务，我们可以考虑使用复杂的（非线性或深度）模型，即加深网络的宽度和深度，提高神经网络的能力。

但是如果网络过于宽和深，就会出现第三张图展示的过拟合的情况。

出现过拟合的原因：

1. 训练集的数量和模型的复杂度不匹配，样本数量级小于模型的参数
2. 训练集和测试集的特征分布不一致
3. 样本噪音大，使得神经网络学习到了噪音，正常样本的行为被抑制
4. 迭代次数过多，过分拟合了训练数据，包括噪音部分和一些非重要特征

既然模型过于复杂，那么我们简化模型不就行了吗？为什么要用复杂度不匹配的模型呢？有两个原因：
1. 因为有的模型以及非常成熟了，比如VGG16，可以不调参而直接用于你自己的数据训练，此时如果你的数据数量不够多，但是又想使用现有模型，就需要给模型加正则项了。
2. 使用相对复杂的模型，可以比较快速地使得网络训练收敛，以节省时间。



###  过拟合例子

我们将要使用MNIST数据集做例子，模拟出令一个过拟合（分类）的情况。从上面的过拟合出现的4点原因分析，第2点和第3点对于MNIST数据集来说并不成立，MNIST数据集有60000个样本，这足以保证它的特征分布的一致性，少数样本的噪音也会被大多数正常的数据所淹没。但是如果我们只选用其中的很少一部分的样本，则特征分布就可能会有偏差，而且独立样本的噪音会变得突出一些。

再看过拟合原因中的第1点和第4点，我们利用第14章中的已有知识和代码，搭建一个复杂网络很容易，而且迭代次数完全可以由代码来控制。

首先，只使用1000个样本来做训练，如下面的代码所示，调用一个ReadLessData(1000)函数，并且用GenerateValidationSet(k=10)函数把1000个样本分成900和100两部分，分别做为训练集和验证集：

```Python
def LoadData():
    mdr = MnistImageDataReader(train_image_file, train_label_file, test_image_file, test_label_file, "vector")
    mdr.ReadLessData(1000)
    mdr.Normalize()
    mdr.GenerateDevSet(k=10)
    return mdr
```

然后，我们搭建一个深度网络：

<img src="./media/7/overfit_net.png" />

这个网络有5个全连接层，前4个全连接层后接ReLU激活函数层，最后一个全连接层接Softmax分类函数做10分类。由于我们在第14章就已经搭建好了深度神经网络的Mini框架，所以可以简单地搭建这个网络，如下代码所示：

```Python
def Net(dateReader, num_input, num_hidden, num_output, params):
    net = NeuralNet(params)

    fc1 = FcLayer(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    relu1 = ActivatorLayer(Relu())
    net.add_layer(relu1, "relu1")

    fc2 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc2, "fc2")
    relu2 = ActivatorLayer(Relu())
    net.add_layer(relu2, "relu2")

    fc3 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc3, "fc3")
    relu3 = ActivatorLayer(Relu())
    net.add_layer(relu3, "relu3")

    fc4 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc4, "fc4")
    relu4 = ActivatorLayer(Relu())
    net.add_layer(relu4, "relu4")

    fc5 = FcLayer(num_hidden, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ActivatorLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=1)
    
    net.ShowLossHistory()
```

net.train(dataReader, checkpoint=1)函数的参数checkpoint的含义是，每隔1个epoch记录一次训练过程中的损失值和准确率。可以设置成大于1的数字，比如10，意味着每10个epoch检查一次。也可以设置为小于1大于0的数比如0.5，假设在一个epoch中要迭代100次，则每50次检查一次。

在main过程中，设置一些超参数，然后调用刚才建立的Net进行训练：

```Python
if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden = 30
    num_output = 10
    max_epoch = 200
    batch_size = 100
    learning_rate = 0.1
    eps = 1e-5

    params = CParameters(
      learning_rate, max_epoch, batch_size, eps,
      LossFunctionName.CrossEntropy3, 
      InitialMethod.Xavier, 
      OptimizerName.SGD)

    Net(dataReader, num_input, num_hidden, num_hidden, num_hidden, num_hidden, num_output, params)
```

在超参数中，我们指定了：
1. 每个隐层30个神经元（4个隐层在Net函数里指定）
2. 最多训练200个epoch
3. 批大小为100个样本
4. 学习率为0.1
5. 多分类交叉熵损失函数(CrossEntropy3)
6. Xavier权重初始化方法
7. 随机梯度下降算法


在训练集上（蓝色曲线），很快就达到了损失函数值趋近于0，准确度100%的程度。而在验证集上（红色曲线），损失函数值缺越来越大，准确度也在下降。这就造成了一个典型的过拟合网络，即所谓U型曲线，无论是损失函数值和准确度，都呈现出了这种分化的特征。

我们再看打印输出部分：
```
epoch=199, total_iteration=1799
loss_train=0.0015, accuracy_train=1.000000
loss_valid=0.9956, accuracy_valid=0.860000
time used: 5.082462787628174
total weights abs sum= 1722.470655813152
total weights = 26520
little weights = 2815
zero weights = 27
testing...
rate=8423 / 10000 = 0.8423
```

结果说明：

1. 第199个epoch上（从0开始计数，所以实际是第200个epoch），训练集的损失为0.0015，准确率为100%。测试集损失值0.9956，准确率86%。过拟合线性很严重。
2. total weights abs sum = 1722.4706，实际上是把所有全连接层的权重值先取绝对值，再求和。这个值和下面三个值在后面会有比较说明。
3. total weights = 26520，一共26520个权重值，偏移值不算在内。
4. little weights = 2815，一共2815个权重值小于0.01。
5. zero weights = 27，是权重值中接近于0的数量（小于0.0001）。
6. 测试准确率为84.23%

在着手解决过拟合的问题之前，我们先来学习一下关于偏差与方差的知识，以便得到一些理论上的指导，虽然神经网络是一门实验学科。

### 解决过拟合问题

有了直观感受和理论知识，下面我们看看如何解决过拟合问题：

1. 数据扩展
2. 正则
3. 丢弃法
4. 早停法
5. 集成学习法
6. 特征工程（属于传统机器学习范畴，不在此处讨论）
7. 简化模型，减小网络的宽度和深度


## 直观的解释
## 神经网络训练的例子

我们在前面讲过数据集的使用，包括训练集、验证集、测试集。在训练过程中，我们要不断监测训练集和验证集在当前模型上的误差，和上面的打靶的例子一样，有可能产生四种情况：

|情况|训练集误差A|验证集误差B|偏差|方差|说明|
|---|---|---|---|---|---|
|情况1|1.5%|1.7%|低偏差|低方差|A和B都很好，适度拟合|
|情况2|12.3%|11.4%|高偏差|低方差|A和B都很不好，欠拟合|
|情况3|1.2%|13.1%|低偏差|高方差|A很好，但B不好，过拟合|
|情况4|12.3%|21.5%|高偏差|高方差|A不好，B更不好，欠拟合|

在本例中，偏差衡量训练集误差，方差衡量训练集误差和验证集误差的比值。

上述四种情况的应对措施：

- 情况1
  
  效果很好，可以考虑进一步降低误差值，提高准确度。

- 情况2

  训练集和验证集同时出现较大的误差，有可能是：迭代次数不够、数据不好、网络设计不好，需要继续训练，观察误差变化情况。

- 情况3

  训练集的误差已经很低了，但验证集误差很高，说明过拟合了，即训练集中的某些特殊样本影响了网络参数，但类似的样本在验证集中并没有出现

- 情况4

  两者误差都很大，目前还看不出来是什么问题，需要继续训练

# 偏差-方差分解

除了用上面的试验来估计泛化误差外，我们还希望在理论上分析其必然性，这就是偏差-方差分解的作用，bias-variance decomposition。

|符号|含义|
|---|---|
|$x$|测试样本|
|$D$|数据集|
|$y$|x的真实标记|
|$y_D$|x在数据集中标记(可能有误差)|
|$f$|从数据集D学习的模型|
|$f_{x;D}$|从数据集D学习的模型对x的预测输出|
|$f_x$|模型f对x的期望预测输出|

学习算法期望的预测：
$$f_x=E[f_{x;D}] \tag{1}$$
不同的训练集/验证集产生的预测方差：
$$var(x)=E[(f_{x;D}-f_x)^2] \tag{2}$$
噪声：
$$\epsilon^2=E[(y_D-y)^2] \tag{3}$$
期望输出与真实标记的偏差：
$$bias^2(x)=(f_x-y)^2 \tag{4}$$
算法的期望泛化误差：
$$
\begin{aligned}
E(f;D)&=E[(f_{x;D}-y_D)^2] \\
&=E[(f_{x;D}-f_x+f_x-y_D)^2] \\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y_D)^2] \\
&+E[2(f_{x;D}-f_x)(f_x-y_D)](从公式1，此项为0) \\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y_D)^2] \\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y+y-y_D)^2] \\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y)^2]+E(y-y_D)^2] \\
&+E[2(f_x-y)(y-y_D)](噪声期望为0，所以此项为0)\\
&=E[(f_{x;D}-f_x)^2]+(f_x-y)^2+E[(y-y_D)^2] \\
&=var(x) + bias^2(x) + \epsilon^2
\end{aligned}
$$

所以，各个项的含义是：

- 偏差：度量了学习算法的期望与真实结果的偏离程度，即学习算法的拟合能力。
- 方差：训练集与验证集的差异造成的模型表现的差异。
- 噪声：当前数据集上任何算法所能到达的泛化误差的下线，即学习问题本身的难度。

想当然地，我们希望偏差与方差越小越好，但实际并非如此。一般来说，偏差与方差是有冲突的，称为偏差-方差窘境 (bias-variance dilemma)。

- 给定一个学习任务，在训练初期，由于训练不足，网络的拟合能力不够强，偏差比较大，也是由于拟合能力不强，数据集的特征也无法使网络产生显著变化，也就是欠拟合的情况。
- 随着训练程度的加深，网络的拟合能力逐渐增强，训练数据的特征也能够渐渐被网络学到。
- 充分训练后，网络的拟合能力已非常强，训练数据的微小特征都会导致网络发生显著变化，当训练数据自身的、非全局的特征被网络学到了，则将发生过拟合。


在上图中，随着训练程度的增加，偏差（蓝色实线）一路下降，但是方差（蓝色虚线）一路上升，整体误差（红色实线，偏差+方差+噪音误差）呈U形，最佳平衡点就是U形的最低点。

## 没有免费午餐定理

没有免费午餐定理（No Free Lunch Theorem，NFL）是由Wolpert和Macerday在最优化理论中提出的。没有免费午餐定理证明：对于基于迭代的最优化算法，不存在某种算法对所有问题（有限的搜索空间内）都有效。如果一个算法对某些问题有效，那么它一定在另外一些问题上比纯随机搜索算法更差。

还可以理解为在所有可能的数据生成分布上平均之后，每一个分类算法在未事先观测的点上都有相同的错误率。也就是说，不能脱离具体问题来谈论算法的优劣，任何算法都有局限性。必须要“具体问题具体分析”。