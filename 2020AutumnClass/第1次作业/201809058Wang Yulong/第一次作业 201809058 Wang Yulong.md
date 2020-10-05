<font face="楷体">201809058  Wang Yulong</font>


<font face="楷体" font color=pink><center>**1.人工智能**</font> </center>
=======
<font face="楷体" font color=pink>**1.1 定义：**</font> <font face="楷体" >如果程序解某种任务T的效果P随着经验E的增加而得到了提高，
那么这个程序就能从经验E中学到了关于任务T的知识，并让效果P得到提高。</font>

<font face="楷体" font color=#00FFFF size=4>1.2 **机器学习的具体过程:**</font>
-------
<font face="楷体" >·选择一个模型结构（例如逻辑回归、决策树等），构建上文中的程序</font>

<font face="楷体">·将训练数据（包含输入和输出）输入模型中，不断学习，得到经验E</font>

<font face="楷体">·通过不断执行任务T并衡量效果P，让P不断提高，直到达到一个满意的值</font>


<font face="楷体" font color=#8FBC8F>1.3 BP神经元模型</font>
----------------
&emsp;&emsp;<font face="楷体" >在生物神经网络中，每个神经元与其他神经元相连，当它兴奋时，就会像相邻的神经元发送化学物质，从而改变这些神经元内的电位；如果某神经元的电位超过了一个阈值，那么它就会被激活（兴奋），向其他神经元发送化学物质。把许多这样的神经元按照一定的层次结构连接起来，我们就构建了一个神经网络。</font>


![BP神经元结构图](https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=3436986209,975932292&fm=26&gp=0.jpg)
<font face="楷体" font color=#BDB76B><center>**BP神经元结构图**</font></center>


---
#  <font face="楷体" color=#00BFFF><center>**2.神经元的基本训练**</font></center>


<font face="楷体" color=red>**基本思想：**</font><font face="楷体">首先人为设定一个初始值a【初始化】=>观察初始值a和训练集中y值的偏差【找偏差】=>修改方案，力求机器的值和真实值y的偏差越来越小，直至在误差允许范围内【机器学习】</font>


## <font face="楷体">2.1概念梳理</font>

### <font face="楷体">反向传播</font>
<font face="宋体">定义：将输出值反馈到输入.</font>

### <font face="楷体">梯度下降</font>
<font face="宋体">定义：根据损失函数的值，调整机器学习的策略，从而接近真实值.</font>

### <font face="楷体">损失函数</font>
<font face="宋体">定义：机器算出的值与真实值之间的误差.</font>

````Python
线性反向传播---Python
import numpy as np

def target_function(w,b):
    x = 2*w+3*b
    y=2*b+1
    z=x*y
    return x,y,z

def single_variable(w,b,t):
    print("\nsingle variable: b ----- ")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break
        delta_b = delta_z /63
        print("delta_b=%f"%delta_b)
        b = b - delta_b

    print("done!")
    print("final b=%f"%b)

    if __name__ == '__main__':
    w = 3
    b = 4
    t = 150
    single_variable(w,b,t)
````
```Python
测试结果：

single variable: b ----- 
w=3.000000,b=4.000000,z=162.000000,delta_z=12.000000
delta_b=0.190476
w=3.000000,b=3.809524,z=150.217687,delta_z=0.217687 
delta_b=0.003455
w=3.000000,b=3.806068,z=150.007970,delta_z=0.007970 
delta_b=0.000127
w=3.000000,b=3.805942,z=150.000294,delta_z=0.000294
delta_b=0.000005
w=3.000000,b=3.805937,z=150.000011,delta_z=0.000011
delta_b=0.000000
w=3.000000,b=3.805937,z=150.000000,delta_z=0.000000
done!
final b=3.805937
````
$$此为线性反向传播的代码实例，w=3,b=4,t=150,目的是将z值变为目标值t，方法即为在不改变w的前提下，通过改变b值从而达到改变w值的目标,在这个过程中就需要使用偏导数$$
$$   x = 2*w+3*b $$
$$   y=2*b+1   $$  
$$   z=x*y     $$
$$\frac{\partial{Z}}{\partial{w}} =\frac{\partial{Z}}{\partial{x}}·\frac{\partial{x}}{\partial{w}}+\frac{\partial{Z}}{\partial{y}}·\frac{\partial{Z}}{\partial{w}}$$