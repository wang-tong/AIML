## 第一天学习

### 反向传播与梯度下降

反向传播与梯度下降的基本工作原理：

1. 初始化
2. 正向计算
3. 损失函数为我们提供了计算损失的方法
4. 梯度下降是在损失函数基础上向着损失最小的点靠近而指引了网络权重调整的方向
5. 反向传播把损失值反向传给神经网络的每一层，让每一层都根据损失值反向调整权重
6. goto 2，直到精度足够好（比如损失函数值小于0.001）


### 线性反向传播
假设w = 2  b = 3 t = 80   
x =2*w+3*b
y=2*b+1
z=x*y
没有在迭代中重新计算Δb的贡献值
```
def single_variable(w,b，t):
    print("\nsingle variable: b ----- ")
      error = 1e-5
    while(True):
     x,y,z = target_function(w,b)
         delta_z = z - t
         print("w=%f,b=%z=%f,delta_z=%f"%(w,b,z,delta_z))
         if abs(delta_z) < error:
                        break
         delta_b = delta_z /63
         print("delta_b=%f"%delta_b)
                    b = b - delta_b
print("done!")
print("final b=%f"%b) 
```
结果：
``` single variable: b ----- 
w=2.000000,b=3.000000,z=91.000000,delta_z=11.000000
delta_b=0.174603
w=2.000000,b=2.825397,z=82.976568,delta_z=2.976568
delta_b=0.047247
w=2.000000,b=2.778150,z=80.868342,delta_z=0.868342
delta_b=0.013783
w=2.000000,b=2.764367,z=80.258365,delta_z=0.258365
delta_b=0.004101
w=2.000000,b=2.760265,z=80.077313,delta_z=0.077313
delta_b=0.001227
w=2.000000,b=2.759038,z=80.023175,delta_z=0.023175
delta_b=0.000368
w=2.000000,b=2.758670,z=80.006950,delta_z=0.006950
delta_b=0.000110
w=2.000000,b=2.758560,z=80.002085,delta_z=0.002085
delta_b=0.000033
w=2.000000,b=2.758527,z=80.000625,delta_z=0.000625
delta_b=0.000010
w=2.000000,b=2.758517,z=80.000188,delta_z=0.000188
delta_b=0.000003
w=2.000000,b=2.758514,z=80.000056,delta_z=0.000056
delta_b=0.000001
w=2.000000,b=2.758513,z=80.000017,delta_z=0.000017
delta_b=0.000000
w=2.000000,b=2.758513,z=80.000005,delta_z=0.000005
done!
final b=2.758513
```
在每次迭代中都重新计算Δb的贡献值：

结果
```
single variable new: b ----- 
w=2.000000,b=3.000000,z=91.000000,delta_z=11.000000
factor_b=47.000000, delta_b=0.234043
w=2.000000,b=2.765957,z=80.328656,delta_z=0.328656
factor_b=44.191489, delta_b=0.007437
w=2.000000,b=2.758520,z=80.000332,delta_z=0.000332
factor_b=44.102244, delta_b=0.000008
w=2.000000,b=2.758513,z=80.000000,delta_z=0.000000
done!
final b=2.758513
```
没有在迭代中重新计算Δb,Δw的贡献值
结果：
```
double variable: w, b -----
w=2.000000,b=3.000000,z=91.000000,delta_z=11.000000
delta_b=0.087302, delta_w=0.305556
w=1.694444,b=2.912698,z=82.771479,delta_z=2.771479
delta_b=0.021996, delta_w=0.076986
w=1.617459,b=2.890703,z=80.746363,delta_z=0.746363
delta_b=0.005924, delta_w=0.020732
w=1.596727,b=2.884779,z=80.204304,delta_z=0.204304
delta_b=0.001621, delta_w=0.005675
w=1.591051,b=2.883158,z=80.056170,delta_z=0.056170
delta_b=0.000446, delta_w=0.001560
w=1.589491,b=2.882712,z=80.015461,delta_z=0.015461
delta_b=0.000123, delta_w=0.000429
w=1.589062,b=2.882589,z=80.004257,delta_z=0.004257
delta_b=0.000034, delta_w=0.000118
w=1.588943,b=2.882555,z=80.001172,delta_z=0.001172
delta_b=0.000009, delta_w=0.000033
w=1.588911,b=2.882546,z=80.000323,delta_z=0.000323
delta_b=0.000003, delta_w=0.000009
w=1.588902,b=2.882543,z=80.000089,delta_z=0.000089
delta_b=0.000001, delta_w=0.000002
w=1.588899,b=2.882543,z=80.000024,delta_z=0.000024
delta_b=0.000000, delta_w=0.000001
w=1.588899,b=2.882543,z=80.000007,delta_z=0.000007
done!
final b=2.882543
final w=1.588899
```

在每次迭代中都重新计算Δb,Δw的贡献值(factor_b和factor_w每次都变化)：
结果：
```
double variable new: w, b -----
w=2.000000,b=3.000000,z=91.000000,delta_z=11.000000
factor_b=47.000000, factor_w=14.000000, delta_b=0.117021, delta_w=0.392857
w=1.607143,b=2.882979,z=80.266054,delta_z=0.266054
factor_b=44.024316, factor_w=13.531915, delta_b=0.003022, delta_w=0.009831
w=1.597312,b=2.879957,z=80.000174,delta_z=0.000174
factor_b=43.948733, factor_w=13.519828, delta_b=0.000002, delta_w=0.000006
w=1.597306,b=2.879955,z=80.000000,delta_z=0.000000
done!
final b=2.879955
final w=1.597306
```

#### 非线性反向传播
运行结果：
```  
how to play: 1) input x, 2) calculate c, 3) input target number but not faraway from c
input x as initial number(1.2,10), you can try 1.3:
3
c=1.482304
input y as target number(0.5,2), you can try 1.8:
1.5
forward...
x=3.000000,a=9.000000,b=2.197225,c=1.482304
backward...
delta_c=-0.017696, delta_b=-0.052462, delta_a=-0.472160, delta_x=-0.078693

forward...
x=3.078693,a=9.478353,b=2.249011,c=1.499670
backward...
done!
```
该组合函数图像（蓝色）和导数图像（绿色）：

![](./media/1.jpg)


## 总结
通过今天在GitHub上和mooc的学习，对神经网络有了初步的认识，代码也都跑了一遍，一些细致的地方还没有搞懂，总体上还是感觉不错，天天宅在加里也有事可做，感觉还行。