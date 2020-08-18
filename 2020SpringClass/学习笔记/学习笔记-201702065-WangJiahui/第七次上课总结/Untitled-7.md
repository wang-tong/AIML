# 第七次上课总结
## 第十五章 网络优化
随着网络的加深，训练变得越来越困难，时间越来越长，原因可能是：
- 参数多
- 数据量大
- 梯度消失
- 损失函数坡度平缓

为了解决上面这些问题，科学家们在深入研究网络表现的前提下，发现在下面这些方向上经过一些努力，可以给深度网络的训练带来或多或少的改善：
- 权重矩阵初始化
- 批量归一化
- 梯度下降优化算法
- 自适应学习率算法
### 权重矩阵初始化
权重矩阵初始化是一个非常重要的环节，是训练神经网络的第一步，选择正确的初始化方法会带了事半功倍的效果。这就好比攀登喜马拉雅山，如果选择从南坡登山，会比从北坡容易很多。而初始化权重矩阵，相当于下山时选择不同的道路，在选择之前并不知道这条路的难易程度，只是知道它可以抵达山下。这种选择是随机的，即使你使用了正确的初始化算法，每次重新初始化时也会给训练结果带来很多影响。

比如第一次初始化时得到权重值为(0.12847，0.36453)，而第二次初始化得到(0.23334，0.24352)，经过试验，第一次初始化用了3000次迭代达到精度为96%的模型，第二次初始化只用了2000次迭代就达到了相同精度。这种情况在实践中是常见的。
#### 零初始化

即把所有层的W值的初始值都设置为0。

$$
W = 0
$$
#### 随机初始化

把W初始化均值为0，方差为1的矩阵：

$$
W \sim G \begin{bmatrix} 0, 1 \end{bmatrix}
$$
#### Xavier初始化方法

条件：正向传播时，激活值的方差保持不变；反向传播时，关于状态值的梯度的方差保持不变。

$$
W \sim U \begin{bmatrix} -\sqrt{{6 \over n_{input} + n_{output}}}, \sqrt{{6 \over n_{input} + n_{output}}} \end{bmatrix}
$$

假设激活函数关于0对称，且主要针对于全连接神经网络。适用于tanh和softsign。

即权重矩阵参数应该满足在该区间内的均匀分布。其中的W是权重矩阵，
#### MSRA初始化方法

又叫做He方法，因为作者姓何。

条件：正向传播时，状态值的方差保持不变；反向传播时，关于激活值的梯度的方差保持不变。
#### 运行结果
![](\42.png)
![](\43.png)
![](\44.png)
![](\45.png)
![](\46.png)
### 梯度下降优化算法
#### 输入和参数

- $\eta$ - 全局学习率

#### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

更新参数：$\theta_t = \theta_{t-1}  - \eta \cdot g_t$

---

随机梯度下降算法，在当前点计算梯度，根据学习率前进到下一点。到中点附近时，由于样本误差或者学习率问题，会发生来回徘徊的现象，很可能会错过最优解。
#### 动量算法 Momentum

SGD方法的一个缺点是其更新方向完全依赖于当前batch计算出的梯度，因而十分不稳定，因为数据有噪音。

Momentum算法借用了物理中的动量概念，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向。这样一来，可以在一定程度上增加稳定性，从而学习地更快，并且还有一定摆脱局部最优的能力。Momentum算法会观察历史梯度，若当前梯度的方向与历史梯度一致（表明当前样本不太可能为异常点），则会增强这个方向的梯度。若当前梯度与历史梯方向不一致，则梯度会衰减。
### 梯度加速算法 NAG

Nesterov Accelerated Gradient，或者叫做Nesterov Momentum。

在小球向下滚动的过程中，我们希望小球能够提前知道在哪些地方坡面会上升，这样在遇到上升坡面之前，小球就开始减速。这方法就是Nesterov Momentum，其在凸优化中有较强的理论保证收敛。并且，在实践中Nesterov Momentum也比单纯的Momentum 的效果好。

#### 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，缺省取值0.9
- v - 动量，初始值为0
  
#### 算法

---

临时更新：$\hat \theta = \theta_{t-1} - \alpha \cdot v_{t-1}$

前向计算：$f(\hat \theta)$

计算梯度：$g_t = \nabla_{\hat\theta} J(\hat \theta)$

计算速度更新：$v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$

更新参数：$\theta_t = \theta_{t-1}  - v_t$

---
### 运行结果
![](\47.png)
![](\48.png)
![](\49.png)
![](\50.png)
![](\51.png)
![](\52.png)
![](\53.png)
![](\54.png)
### 自适应学习率算法

#### AdaGrad

Adaptive subgradient method.

AdaGrad是一个基于梯度的优化算法，它的主要功能是：它对不同的参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。因此，他很适合于处理稀疏数据。

在这之前，我们对于所有的参数使用相同的学习率进行更新。但 Adagrad 则不然，对不同的训练迭代次数t，AdaGrad 对每个参数都有一个不同的学习率。这里开方、除法和乘法的运算都是按元素运算的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

##### 输入和参数

- $\eta$ - 全局学习率
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-6
- r = 0 初始值
  
##### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

累计平方梯度：$r_t = r_{t-1} + g_t \odot g_t$

计算梯度更新：$\Delta \theta = {\eta \over \epsilon + \sqrt{r_t}} \odot g_t$

更新参数：$\theta_t=\theta_{t-1} - \Delta \theta$

---
#### AdaDelta

Adaptive Learning Rate Method.

AdaDelta法是AdaGrad 法的一个延伸，它旨在解决它学习率不断单调下降的问题。相比计算之前所有梯度值的平方和，AdaDelta法仅计算在一个大小为w的时间区间内梯度值的累积和。

但该方法并不会存储之前梯度的平方值，而是将梯度值累积值按如下的方式递归地定义：关于过去梯度值的衰减均值，当前时间的梯度均值是基于过去梯度均值和当前梯度值平方的加权平均，其中是类似上述动量项的权值。

##### 输入和参数

- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-5
- $\alpha \in [0,1)$ - 衰减速率，建议0.9
- s - 累积变量，初始值0
- r - 累积变量变化量，初始为0
 
##### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

累积平方梯度：$s_t = \alpha \cdot s_{t-1} + (1-\alpha) \cdot g_t \odot g_t$

计算梯度更新：$\Delta \theta = \sqrt{r_{t-1} + \epsilon \over s_t + \epsilon} \odot g_t$

更新梯度：$\theta_t = \theta_{t-1} - \Delta \theta$

更新变化量：$r = \alpha \cdot r_{t-1} + (1-\alpha) \cdot \Delta \theta \odot \Delta \theta$

---
#### 均方根反向传播 RMSProp

Root Mean Square Prop。

RMSprop 是由 Geoff Hinton 在他 Coursera 课程中提出的一种适应性学习率方法，至今仍未被公开发表。RMSprop法要解决AdaGrad的学习率缩减问题。

##### 输入和参数

- $\eta$ - 全局学习率，建议设置为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\alpha$ - 衰减速率，建议缺省取值0.9
- $r$ - 累积变量矩阵，与$\theta$尺寸相同，初始化为0
  
##### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

累计平方梯度：$r = \alpha \cdot r + (1-\alpha)(g_t \odot g_t)$

计算梯度更新：$\Delta \theta = {\eta \over \sqrt{r + \epsilon}} \odot g_t$

更新参数：$\theta_{t}=\theta_{t-1} - \Delta \theta$

---
#### Adam - Adaptive Moment Estimation

计算每个参数的自适应学习率，相当于RMSProp + Momentum的效果，Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均。和AdaGrad算法、RMSProp算法以及AdaDelta算法一样，目标函数自变量中每个元素都分别拥有自己的学习率。

##### 输入和参数

- t - 当前迭代次数
- $\eta$ - 全局学习率，建议缺省值为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\beta_1, \beta_2$ - 矩估计的指数衰减速率，$\in[0,1)$，建议缺省值分别为0.9和0.999

##### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

计数器加一：$t=t+1$

更新有偏一阶矩估计：$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$

更新有偏二阶矩估计：$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2)(g_t \odot g_t)$

修正一阶矩的偏差：$\hat m_t = m_t / (1-\beta_1^t)$

修正二阶矩的偏差：$\hat v_t = v_t / (1-\beta_2^t)$

计算梯度更新：$\Delta \theta = \eta \cdot \hat m_t /(\epsilon + \sqrt{\hat v_t})$

更新参数：$\theta_t=\theta_{t-1} - \Delta \theta$

---
### 运行结果
![](\55.png)
![](\56.png)
![](\57.png)
![](\58.png)
![](\59.png)
![](\60.png)
![](\61.png)
![](\62.png)
![](\63.png)
![](\64.png)
![](\65.png)
![](\66.png)
![](\67.png)
![](\68.png)
![](\69.png)
![](\70.png)
![](\71.png)
![](\72.png)
### 算法在等高线图上的效果比较

#### 模拟效果比较

为了简化起见，我们先用一个简单的二元二次函数来模拟损失函数的等高线图，测试一下我们在前面实现的各种优化器。但是以下测试结果只是一个示意性质的，可以理解为在绝对理想的条件下（样本无噪音，损失函数平滑等等）的各算法的表现。

$$z = {x^2 \over 10} + y^2 \tag{1}$$
#### 运行结果
![](\73.png)
![](\74.png)
### 批量归一化的原理

有的书翻译成归一化，有的翻译成正则化，英文Batch Normalization，简称为BatchNorm，或BN。

#### 基本数学知识

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
#### 前向计算

##### 符号表

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

##### 计算图（示意）
$X1,X2,X3$表示三个样本（实际上一般用32，64这样的批大小），我们假设每个样本只有一个特征值（否则X将会是一个样本数乘以特征值数量的矩阵）。
1. 先从一堆X中计算出$\mu_B$；
2. 再用X和$\mu_B$计算出$\sigma_B$；
3. 再用X和$\mu_B$、$\sigma_B$计算出$n_i$，每个x对应一个n；
4. 最后用$\gamma 和 \beta$，把n转换成z，每个z对应一个n。

#### 测试和推理时的归一化方法

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

#### 批量归一化的优点

1. 可以选择比较大的初始学习率，让你的训练速度提高。
   
    以前还需要慢慢调整学习率，甚至在网络训练到一定程度时，还需要想着学习率进一步调小的比例选择多少比较合适，现在我们可以采用初始很大的学习率，因为这个算法收敛很快。当然这个算法即使你选择了较小的学习率，也比以前的收敛速度快，因为它具有快速训练收敛的特性；

2. 减少对初始化的依赖
   
    一个不太幸运的初始化，可能会造成网络训练实际很长，甚至不收敛。

3. 减少对正则的依赖
   
   在第16章中，我们将会学习正则化知识，以增强网络的泛化能力。采用BN算法后，我们会逐步减少对正则的依赖，比如令人头疼的dropout、L2正则项参数的选择问题，或者可以选择更小的L2正则约束参数了，因为BN具有提高网络泛化能力的特性；
#### 运行结果
![](\75.png)
![](\76.png)
![](\77.png)
![](\78.png)
![](\79.png)
### 批量归一化的实现

在这一节中，我们将会动手实现一个批量归一化层，来验证BN的实际作用。

#### 反向传播

在上一节中，我们知道了BN的正向计算过程，这一节中，为了实现完整的BN层，我们首先需要推导它的反向传播公式，然后用代码实现。本节中的公式序号接上一节，以便于说明。

首先假设已知从上一层回传给BN层的误差矩阵是：

$$\delta = {dJ \over dZ}，\delta_i = {dJ \over dz_i} \tag{10}$$

##### 求BN层参数梯度

则根据公式9，求$\gamma 和 \beta$的梯度：

$${dJ \over d\gamma} = \sum_{i=1}^m {dJ \over dz_i}{dz_i \over d\gamma}=\sum_{i=1}^m \delta_i \cdot n_i \tag{11}$$

$${dJ \over d\beta} = \sum_{i=1}^m {dJ \over dz_i}{dz_i \over d\beta}=\sum_{i=1}^m \delta_i \tag{12}$$

注意$\gamma$和$\beta$的形状与批大小无关，只与特征值数量有关，我们假设特征值数量为1，所以它们都是一个标量。在从计算图看，它们都与N,Z的全集相关，而不是某一个样本，因此会用求和方式计算。

##### 求BN层的前传误差矩阵

下述所有乘法都是element-wise的矩阵点乘，不再特殊说明。

从正向公式中看，对z有贡献的数据链是：

- $z_i \leftarrow n_i \leftarrow x_i$
- $z_i \leftarrow n_i \leftarrow \mu_B \leftarrow x_i$
- $z_i \leftarrow n_i \leftarrow \sigma^2_B \leftarrow x_i$
- $z_i \leftarrow n_i \leftarrow \sigma^2_B \leftarrow \mu_B \leftarrow x_i$

从公式8，9：

$$
{dJ \over dx_i} = {dJ \over d n_i}{d n_i \over dx_i} + {dJ \over d \sigma^2_B}{d \sigma^2_B \over dx_i} + {dJ \over d \mu_B}{d \mu_B \over dx_i} \tag{13}
$$

公式13的右侧第一部分（与全连接层形式一样）：

$$
{dJ \over d n_i}=  {dJ \over dz_i}{dz_i \over dn_i} = \delta_i \cdot \gamma\tag{14}
$$

上式等价于：

$$
{dJ \over d N}= \delta \cdot \gamma\tag{14}
$$

公式14中，我们假设样本数为64，特征值数为10，则得到一个64x10的结果矩阵（因为1x10的矩阵会被广播为64x10的矩阵）：

$$\delta^{(64 \times 10)} \odot \gamma^{(1 \times 10)}=R^{(64 \times 10)}$$

公式13的右侧第二部分，从公式8：
$$
{d n_i \over dx_i}={1 \over \sqrt{\sigma^2_B + \epsilon}} \tag{15}
$$

公式13的右侧第三部分，从公式8（注意$\sigma^2_B$是个标量，而且与X,N的全集相关，要用求和方式）：

$$
{dJ \over d \sigma^2_B} = \sum_{i=1}^m {dJ \over d n_i}{d n_i \over d \sigma^2_B} 
$$
$$
= -{1 \over 2}(\sigma^2_B + \epsilon)^{-3/2}\sum_{i=1}^m {dJ \over d n_i} \cdot (x_i-\mu_B) \tag{16}
$$

公式13的右侧第四部分，从公式7：
$$
{d \sigma^2_B \over dx_i} = {2(x_i - \mu_B) \over m} \tag{17}
$$

公式13的右侧第五部分，从公式7，8：

$$
{dJ \over d \mu_B}={dJ \over d n_i}{d n_i \over d \mu_B} + {dJ \over  d\sigma^2_B}{d \sigma^2_B \over d \mu_B} \tag{18}
$$

公式18的右侧第二部分，根据公式8：

$$
{d n_i \over d \mu_B}={-1 \over \sqrt{\sigma^2_B + \epsilon}} \tag{19}
$$

公式18的右侧第四部分，根据公式7（$\sigma^2_B和\mu_B$与全体$x_i$相关，所以要用求和）：

$$
{d \sigma^2_B \over d \mu_B}=-{2 \over m}\sum_{i=1}^m (x_i- \mu_B) \tag{20}
$$

所以公式18是：

$$
{dJ \over d \mu_B}=-{\delta \cdot \gamma \over \sqrt{\sigma^2_B + \epsilon}} - {2 \over m}{dJ \over d \sigma^2_B}\sum_{i=1}^m (x_i- \mu_B) \tag{18}
$$

公式13的右侧第六部分，从公式6：

$$
{d \mu_B \over dx_i} = {1 \over m} \tag{21}
$$

所以，公式13最后是这样的：

$$
{dJ \over dx_i} = {\delta \cdot \gamma \over \sqrt{\sigma^2_B + \epsilon}} + {dJ \over d\sigma^2_B} \cdot {2(x_i - \mu_B) \over m} + {dJ \over d\mu_B} \cdot {1 \over m} \tag{13}
$$

#### 运行结果
![](\80.png)