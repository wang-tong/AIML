# Step5 - NonLinearClassification
# 多入单出的双层神经网络
## 双变量非线性二分类
###  二分类模型的评估标准
### 准确率 Accuracy

也可以称之为精度，我们在本书中混用这两个词。

对于二分类问题，假设测试集上一共1000个样本，其中550个正例，450个负例。测试一个模型时，得到的结果是：521个正例样本被判断为正类，435个负例样本被判断为负类，则正确率计算如下：

$$(521+435)/1000=0.956$$

即正确率为95.6%。这种方式对多分类也是有效的，即三类中判别正确的样本数除以总样本数，即为准确率。

但是这种计算方法丢失了很多细节，比如：是正类判断的精度高还是负类判断的精度高呢？因此，我们还有如下一种评估标准。

### 混淆矩阵

还是用上面的例子，如果具体深入到每个类别上，会分成4部分来评估：
- 正例中被判断为正类的样本数（TP-True Positive）：521
- 正例中被判断为负类的样本数（FN-False Negative）：550-521=29
- 负例中被判断为负类的样本数（TN-True Negative）：435
- 负例中被判断为正类的样本数（FP-False Positive）：450-435=15

用矩阵表达的话是这样的：

|预测值|被判断为正类|被判断为负类|Total|
|---|---|---|---|
|样本实际为正例|TP-True Positive|FN-False Negative|Actual Positive=TP+FN|
|样本实际为负例|FP-False Positive|TN-True Negative|Actual Negative=FP+TN|
|Total|Predicated Postivie=TP+FP|Predicated Negative=FN+TN|

从混淆矩阵中可以得出以下统计指标：

- 准确率 Accuracy

$$Accuracy = {TP+TN \over TP+TN+FP+FN}$$
$$={521+435 \over 521+29+435+15}=0.956$$

这个指标就是上面提到的准确率，越大越好。

- 精确率/查准率 Precision

分子为被判断为正类并且真的是正类的样本数，分母是被判断为正类的样本数。越大越好。

$$Precision={TP \over TP+FP}$$
$$={521 \over 521+15}=0.972$$

- 召回率/查全率 Recall

$$Recall= {TP \over TP+FN}$$
$$={521 \over 521+29}=0.947$$
分子为被判断为正类并且真的是正类的样本数，分母是真的正类的样本数。越大越好。

- TPR - True Positive Rate 真正例率

$$TPR = {TP \over TP + FN}$$
$$=Recall=0.947$$

- FPR - False Positive Rate 假正例率

$$FPR = {FP \over FP+TN}$$
$$={15 \over 15+435}=0.033$$

分子为被判断为正类的负例样本数，分母为所有负类样本数。越小越好。

- 调和平均值 F1

$$F1={2 \times Precision \times Recall \over Precision+Recall}={2 \over {1 / P}+{1 / R}}$$
$$={2 \times 0.972 \times 0.947 \over 0.972+0.947}=0.959$$

该值越大越好。

- ROC曲线与AUC

ROC曲线的横坐标是FPR，纵坐标是TPR。

在二分类器中，如果使用Logistic函数作为分类函数，可以设置一系列不同的阈值，比如[0.1,0.2,0.3...0.9]，把测试样本输入，从而得到一系列的TP、FP、TN、FN，然后就可以绘制如下曲线图。

图中红色的曲线就是ROC曲线，曲线下的面积就是AUC值，区间[0.5,1.0]，面积越大越好。

- ROC曲线越靠近左上角，该分类器的性能越好。
- 对角线表示一个随机猜测分类器。
- 若一个学习器的ROC曲线被另一个学习器的曲线完全包住，则可判断后者性能优于前者。
- 若两个学习器的ROC曲线没有包含关系，则可以判断ROC曲线下的面积，即AUC，谁大谁好。

<img src="./media/5/ROC.png" ch="500" />

当然在实际应用中，取决于阈值的采样间隔，红色曲线不会这么平滑，由于采样间隔会导致该曲线呈阶梯状。

既然已经这么多标准，为什么还要使用ROC和AUC呢？因为ROC曲线有个很好的特性：当测试集中的正负样本的分布变换的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现样本类不平衡，即正负样本比例差距较大，而且测试数据中的正负样本也可能随着时间变化。

### Kappa statics 

Kappa值，即内部一致性系数(inter-rater,coefficient of internal consistency)，是作为评价判断的一致性程度的重要指标。取值在0～1之间。
- Kappa≥0.75两者一致性较好；
- 0.75>Kappa≥0.4两者一致性一般；
- Kappa<0.4两者一致性较差。 

### Mean absolute error 和 Root mean squared error 

平均绝对误差和均方根误差，用来衡量分类器预测值和实际结果的差异，越小越好。

### Relative absolute error 和 Root relative squared error 

相对绝对误差和相对均方根误差，有时绝对误差不能体现误差的真实大小，而相对误差通过体现误差占真值的比重来反映误差大小。

##  非线性二分类实现

###  定义神经网络结构

<img src="./media/5/xor_nn.png" />

- 输入层两个特征值x1, x2
  $$
  X=\begin{pmatrix}
    x_1 & x_2
  \end{pmatrix}
  $$
- 隐层2x2的权重矩阵W1
$$
  W1=\begin{pmatrix}
    w^1_{11} & w^1_{12} \\
    w^1_{21} & w^1_{22} 
  \end{pmatrix}
$$
- 隐层1x2的偏移矩阵B1

$$
  B1=\begin{pmatrix}
    b^1_{1} & b^1_{2}
  \end{pmatrix}
$$

- 隐层由两个神经元构成
$$
Z1=\begin{pmatrix}
  z^1_{1} & z^1_{2}
\end{pmatrix}
$$
$$
A1=\begin{pmatrix}
  a^1_{1} & a^1_{2}
\end{pmatrix}
$$
- 输出层2x1的权重矩阵W2
$$
  W2=\begin{pmatrix}
    w^2_{11} \\
    w^2_{21}  
  \end{pmatrix}
$$

- 输出层1x1的偏移矩阵B2

$$
  B2=\begin{pmatrix}
    b^2_{1}
  \end{pmatrix}
$$

- 输出层有一个神经元使用Logisitc函数进行分类
$$
  Z2=\begin{pmatrix}
    z^2_{1}
  \end{pmatrix}
$$
$$
  A2=\begin{pmatrix}
    a^2_{1}
  \end{pmatrix}
$$


输入特征值可以有很多，隐层单元也可以有很多，输出单元只有一个，且后面要接Logistic分类函数和二分类交叉熵损失函数。

### 前向计算

根据网络结构，我们有了前向计算图：

<img src="./media/5/binary_forward.png" />

#### 第一层

- 线性计算

$$
z^1_{1} = x_{1} w^1_{11} + x_{2} w^1_{21} + b^1_{1}
$$
$$
z^1_{2} = x_{1} w^1_{12} + x_{2} w^1_{22} + b^1_{2}
$$
$$
Z1 = X \cdot W1 + B1
$$

- 激活函数

$$
a^1_{1} = Sigmoid(z^1_{1})
$$
$$
a^1_{2} = Sigmoid(z^1_{2})
$$
$$
A1=\begin{pmatrix}
  a^1_{1} & a^1_{2}
\end{pmatrix}
$$

#### 第二层

- 线性计算

$$
z^2_1 = a^1_{1} w^2_{11} + a^1_{2} w^2_{21} + b^2_{1}
$$
$$
Z2 = A1 \cdot W2 + B2
$$

- 分类函数

$$a^2_1 = Logistic(z^2_1)$$
$$A2 = Logistic(Z2)$$

#### 损失函数

我们把异或问题归类成二分类问题，所以使用二分类交叉熵损失函数：

$$
loss = -y \ln A2 + (1-y) \ln (1-A2) \tag{12}
$$

###  反向传播

#### 求损失函数对输出层的反向误差

对损失函数求导，可以得到损失函数对输出层的梯度值，即上图中的Z2部分。

根据公式15，求A2和Z2的导数（此处A2、Z2都是标量）：

$$
{\partial loss \over \partial Z2}={\partial loss \over \partial A2}{\partial A2 \over \partial Z2}
$$
$$
={A2-y \over A2(1-A2)} \cdot A2(1-A2)
$$
$$
=A2-y => dZ2 \tag{13}
$$

#### 求W2和B2的梯度

$$
{\partial loss \over \partial W2}=\begin{pmatrix}
  {\partial loss \over \partial w^2_{11}} \\
  \\
  {\partial loss \over \partial w^2_{21}}
\end{pmatrix}
=\begin{pmatrix}
  {\partial loss \over \partial Z2}{\partial z2 \over \partial w^2_{11}} \\
  \\
  {\partial loss \over \partial Z2}{\partial z2 \over \partial w^2_{21}}
\end{pmatrix}
$$
$$
=\begin{pmatrix}
  (A2-y)a^1_{1} \\
  (A2-y)a^1_{2} 
\end{pmatrix}
=\begin{pmatrix}
  a^1_{1} \\ a^1_{2}
\end{pmatrix}(A2-y)
$$
$$
=A1^T \cdot dZ2 => dW2  \tag{14}
$$
$${\partial{loss} \over \partial{B2}}=dZ2 => dB2 \tag{15}$$

#### 求损失函数对隐层的反向误差

$$
\frac{\partial{loss}}{\partial{A1}} = \begin{pmatrix}
  {\partial loss \over \partial a^1_{1}} & {\partial loss \over \partial a^1_{2}} 
\end{pmatrix}
$$
$$
=\begin{pmatrix}
\frac{\partial{loss}}{\partial{Z2}} \frac{\partial{Z2}}{\partial{a^1_{1}}} & \frac{\partial{loss}}{\partial{Z2}}  \frac{\partial{Z2}}{\partial{a^1_{2}}}  
\end{pmatrix}
$$
$$
=\begin{pmatrix}
dZ2 \cdot w^2_{11} & dZ2 \cdot w^2_{21}
\end{pmatrix}
$$
$$
=dZ2 \cdot \begin{pmatrix}
  w^2_{11} & w^2_{21}
\end{pmatrix}
$$
$$
=dZ2 \cdot W2^T \tag{16}
$$

$$
{\partial A1 \over \partial Z1}=A1 \odot (1-A1) => dA1\tag{17}
$$

所以最后到达z1的误差矩阵是：

$$
{\partial loss \over \partial Z1}={\partial loss \over \partial A1}{\partial A1 \over \partial Z1}
$$
$$
=dZ2 \cdot W2^T \odot dA1 => dZ1 \tag{18}
$$

有了dZ1后，再向前求W1和B1的误差，就和第5章中一样了，我们直接列在下面：

$$
dW1=X^T \cdot dZ1 \tag{19}
$$
$$
dB1=dZ1 \tag{20}
$$

##  实现逻辑异或门


#### 准备数据

异或数据比较简单，只有4个记录，所以就hardcode在此，不用再建立数据集了。这也给读者一个机会了解如何从DataReader类派生出一个全新的子类XOR_DataReader。

比如在下面的代码中，我们覆盖了父类中的三个方法：

- init() 初始化方法：因为父类的初始化方法要求有两个参数，代表train/test数据文件
- ReadData()方法：父类方法是直接读取数据文件，此处直接在内存中生成样本数据，并且直接令训练集等于原始数据集（不需要归一化），令测试集等于训练集
- GenerateValidationSet()方法，由于只有4个样本，所以直接令验证集等于训练集

因为NeuralNet2中的代码要求数据集比较全，有训练集、验证集、测试集，为了已有代码能顺利跑通，我们把验证集、测试集都设置成与训练集一致，对于解决这个异或问题没有什么影响。



## 实现双弧形二分类

逻辑异或问题的成功解决，可以带给我们一定的信心，但是毕竟只有4个样本，还不能发挥出双层神经网络的真正能力。下面让我们一起来解决问题二，复杂的二分类问题。
### 代码实现

#### 主过程代码

```Python
import numpy as np

from HelperClass2.DataReader import *
from HelperClass2.HyperParameters2 import *
from HelperClass2.NeuralNet2 import *

train_data_name = "../../Data/ch10.train.npz"
test_data_name = "../../Data/ch10.test.npz"

if __name__ == '__main__':
    dataReader = DataReader(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 5, 10000
    eps = 0.08

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Arc_221")
    net.train(dataReader, 5, True)
    net.ShowTrainingTrace()
    
```

此处的代码有几个需要强调的细节：
- n_input = dataReader.num_feature，值为2，而且必须为2，因为只有两个特征值
- n_hidden=2，这是人为设置的隐层神经元数量，可以是大于2的任何整数
- eps精度=0.08是后验知识，笔者通过测试得到的停止条件，用于方便案例讲解
- 网络类型是NetType.BinaryClassifier，指明是二分类网络


# 多入多出的双层神经网络
## 双变量非线性多分类
### 多分类模型的评估标准

我们以三分类问题举例，假设每类有100个样本，一共300个样本，最后的分类结果是：

|样本所属类别|分到类1|分到类2|分到类3|各类样本总数|精(准)确率|
|---|---|---|---|---|---|
|类1|90|4|6|100|90%|
|类2|9|84|5|100|84%|
|类3|1|4|95|100|95%|
|总数|101|93|106|300|89.67%|

- 第1类样本，被错分到2类4个，错分到3类6个，正确90个
- 第2类样本，被错分到1类9个，错分到3类5个，正确84个
- 第3类样本，被错分到1类1个，错分到2类4个，正确95个
 
总体的准确率是89.67%。三类的精确率是90%、84%、95%。

当然也可以计算每个类别的Precision和Recall，但是只在需要时才去做具体计算。比如，当第2类和第3类混淆比较严重时，为了记录模型训练的历史情况，才会把第2类和第3类单独拿出来分析。


# 多入多出的三层神经网络


##  多变量非线性多分类

#### 图片数据归一化

在第5章中，我们学习了数据归一化，下面这段代码是第5章中使用的数据归一化方法：

```Python
def NormalizeData(X):
    X_NEW = np.zeros(X.shape)
    # get number of features
    n = X.shape[0]
    for i in range(n):
        x_row = X[i,:]
        x_max = np.max(x_row)
        x_min = np.min(x_row)
        if x_max != x_min:
            x_new = (x_row - x_min)/(x_max-x_min)
            X_NEW[i,:] = x_new
    return X_NEW
```

下面这段代码是第本章中使用的数据归一化方法：
```Python
    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW
```

我们使用了不一样的归一化数据处理方法，这是为什么呢？

|样本序号|1|2|3|4|...|60000|
|---|---|----|---|--|--|--|
|点1|0|0|0|0|...|0|
|点2|0|0|12|0|...|59|
|点3|0|0|56|253|...|98|
|点m|0|23|148|0|...|0|
|点784|0|0|0|0|...|0|

也就是说，数据虽然分成多个特征值（行），但是每个特征的取值范围实际上都是[0,255]。假设有这样一种情况：

|样本序号|1|2|3|4|...|60000|
|---|---|----|---|--|--|--|
|点1|0|3|0|1|...|0|
|点2|0|0|255|0|...|59|

假设第一行数据中的最大值为3，第二行数据中的最大值为255，如果逐行归一化的话，则3和255都会变成1。它们原本的含义是像素灰度，这样一来本来相差很远的两个数值，都变成1了，这不符合原始含义了。



## 三层神经网络的实现

### 输入层

28x28=784个特征值：

$$
X=\begin{pmatrix}
    x_1 & x_2 & ... & x_{784}
  \end{pmatrix}
$$

#### 隐层1

- 权重矩阵w1形状为784x64

$$
W1=\begin{pmatrix}
    w^1_{1,1} & w^1_{1,2} & ... & w^1_{1,64} \\
    ... & ... & & ... \\
    w^1_{784,1} & w^1_{784,2} & ... & w^1_{784,64} 
  \end{pmatrix}
$$

- 偏移矩阵b1的形状为1x64

$$
B1=\begin{pmatrix}
    b^1_{1} & b^1_{2} & ... & b^1_{64}
  \end{pmatrix}
$$

- 隐层1由64个神经元构成，其结果为1x64的矩阵

$$
Z1=\begin{pmatrix}
    z^1_{1} & z^1_{2} & ... & z^1_{64}
  \end{pmatrix}
$$
$$
A1=\begin{pmatrix}
    a^1_{1} & a^1_{2} & ... & a^1_{64}
  \end{pmatrix}
$$

#### 隐层2

- 权重矩阵w2形状为64x16

$$
W2=\begin{pmatrix}
    w^2_{1,1} & w^2_{1,2} & ... & w^2_{1,16} \\
    ... & ... & & ... \\
    w^2_{64,1} & w^2_{64,2} & ... & w^2_{64,16} 
  \end{pmatrix}
$$

- 偏移矩阵b2的形状是1x16

$$
B2=\begin{pmatrix}
    b^2_{1} & b^2_{2} & ... & b^2_{16}
  \end{pmatrix}
$$

- 隐层2由16个神经元构成

$$
Z2=\begin{pmatrix}
    z^2_{1} & z^2_{2} & ... & z^2_{16}
  \end{pmatrix}
$$
$$
A2=\begin{pmatrix}
    a^2_{1} & a^2_{2} & ... & a^2_{16}
  \end{pmatrix}
$$

#### 输出层

- 权重矩阵w3的形状为16x10

$$
W3=\begin{pmatrix}
    w^3_{1,1} & w^3_{1,2} & ... & w^3_{1,10} \\
    ... & ... & & ... \\
    w^3_{16,1} & w^3_{16,2} & ... & w^3_{16,10} 
  \end{pmatrix}
$$

- 输出层的偏移矩阵b3的形状是1x10

$$
B3=\begin{pmatrix}
    b^3_{1}& b^3_{2} & ... & b^3_{10}
  \end{pmatrix}
$$

- 输出层有10个神经元使用Softmax函数进行分类

$$
Z3=\begin{pmatrix}
    z^3_{1} & z^3_{2} & ... & z^3_{10}
  \end{pmatrix}
$$
$$
A3=\begin{pmatrix}
    a^3_{1} & a^3_{2} & ... & a^3_{10}
  \end{pmatrix}
$$

### 前向计算

我们都是用大写符号的矩阵形式的公式来描述，在每个矩阵符号的右上角是其形状。

#### 隐层1

$$X^{(1,784)} \cdot W1^{(784,64)} + B1^{(1,64)} => Z1^{(1,64)} \tag{1}$$

$$Sigmoid(Z1) => A1^{(1,64)} \tag{2}$$

#### 隐层2

$$A1^{(1,64)} \cdot W2^{(64,16)} + B2^{(1,16)} => Z2^{(1,16)} \tag{3}$$

$$Tanh(Z2) => A2^{(1,16)} \tag{4}$$

#### 输出层

$$A2^{(1,16)} \cdot W3^{(16,10)}  + B3^{(1,10)} => Z3^{(1,10)} \tag{5}$$

$$Softmax(Z3) => A3^{(1,10)} \tag{6}$$

我们的约定是行为样本，列为一个样本的所有特征，这里是784个特征，因为图片高和宽是28x28，总共784个点，把每一个点的值做为特征向量。

两个隐层，分别定义64个神经元和16个神经元。第一个隐层用Sigmoid激活函数，第二个隐层用Tanh激活函数。

输出层10个神经元，再加上一个Softmax计算，最后有a1,a2,...a10十个输出，分别代表0-9的10个数字。

### 反向传播

和以前的两层网络没有多大区别，只不过多了一层，而且用了tanh激活函数，目的是想把更多的梯度值回传，因为tanh函数比sigmoid函数稍微好一些，比如原点对称，零点梯度值大。

#### 输出层

$$dZ3 = A3-Y \tag{7}$$
$$dW3 = A2^T \times dZ3 \tag{8}$$
$$dB3=dZ3 \tag{9}$$

#### 隐层2

$$dA2 = dZ3 \times W3^T \tag{10}$$
$$dZ2 = dA2 \odot (1-A2 \odot A2) \tag{11}$$
$$dW2 = A1^T \times dZ2 \tag{12}$$
$$dB2 = dZ2 \tag{13}$$

#### 隐层1

$$dA1 = dZ2 \times W2^T \tag{14}$$
$$dZ1 = dA1 \odot A1 \odot (1-A1) \tag{15}$$
$$dW1 = X^T \times dZ1 \tag{16}$$
$$dB1 = dZ1 \tag{17}$$

#### 初始化

```Python
class NeuralNet3(object):
    def __init__(self, hp, model_name):
        ...
        self.wb1 = WeightsBias(self.hp.num_input, self.hp.num_hidden1, self.hp.init_method, self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder, False)
        self.wb2 = WeightsBias(self.hp.num_hidden1, self.hp.num_hidden2, self.hp.init_method, self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder, False)
        self.wb3 = WeightsBias(self.hp.num_hidden2, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb3.InitializeWeights(self.subfolder, False)
```
初始化部分需要构造三组WeightsBias对象，请注意各组的输入输出数量，决定了矩阵的形状。

#### 前向计算

```Python
    def forward(self, batch_x):
        # 公式1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        # 公式2
        self.A1 = Sigmoid().forward(self.Z1)
        # 公式3
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        # 公式4
        self.A2 = Tanh().forward(self.Z2)
        # 公式5
        self.Z3 = np.dot(self.A2, self.wb3.W) + self.wb3.B
        # 公式6
        if self.hp.net_type == NetType.BinaryClassifier:
            self.A3 = Logistic().forward(self.Z3)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.A3 = Softmax().forward(self.Z3)
        else:   # NetType.Fitting
            self.A3 = self.Z3
        #end if
        self.output = self.A3
```
前向计算部分增加了一层，并且使用Tanh()做为激活函数。

- 反向传播
```Python
    def backward(self, batch_x, batch_y, batch_a):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]

        # 第三层的梯度输入 公式7
        dZ3 = self.A3 - batch_y
        # 公式8
        self.wb3.dW = np.dot(self.A2.T, dZ3)/m
        # 公式9
        self.wb3.dB = np.sum(dZ3, axis=0, keepdims=True)/m 

        # 第二层的梯度输入 公式10
        dA2 = np.dot(dZ3, self.wb3.W.T)
        # 公式11
        dZ2,_ = Tanh().backward(None, self.A2, dA2)
        # 公式12
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m 
        # 公式13
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m 

        # 第一层的梯度输入 公式8
        dA1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, dA1)
        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m 

    def update(self):
        self.wb1.Update()
        self.wb2.Update()
        self.wb3.Update()
```
反向传播也相应地增加了一层，注意要用对应的Tanh()的反向公式。梯度更新时也是三组权重值同时更新。

- 主过程

```Python
from HelperClass3.MnistImageDataReader import *
from HelperClass3.HyperParameters3 import *
from HelperClass3.NeuralNet3 import *

if __name__ == '__main__':

    dataReader = MnistImageDataReader(mode="vector")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(YNormalizationMethod.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden1 = 64
    n_hidden2 = 16
    n_output = dataReader.num_category
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 40

    hp = HyperParameters3(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet3(hp, "MNIST_64_16")
    net.train(dataReader, 0.5, True)
    net.ShowTrainingTrace(xline="iteration")
```
超参配置：第一隐层64个神经元，第二隐层16个神经元，学习率0.2，批大小128，Xavier初始化，最大训练40个epoch。

# 总结
本次课程我在线下网络看了较多的演示之后做代码练习，仍然出现了诸多问题，最后找同学帮忙解决，并把自己的问题记录了下来加以改正。同时这次课程的的理论知识比较多，通过代码理解知识让我们能更加深刻，多看代码也能增强我们的编程水平。我相信通过我的不懈努力，我的代码编写水平一定会有所提高。学以致用是我的最高期望，我相信在今后的生活与学习中，今日的动手学习能力一定会对我有很大的帮助