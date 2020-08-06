# Step6 - Model Inference
# 神经网络模型概述
首先，我们使用代码生成一个非常简单的深度学习模型，由一个卷积层，一个relu激活函数组成，

```python
import tensorflow as tf
SEED = 46
def main():
    # first generate weights and bias used in conv layers
    conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, 3, 8],  # 5x5 filter, depth 8.
                          stddev=0.1,
                          seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([8]))

    # data and out is placeholder used for input and output data in practice
    data = tf.placeholder(dtype=tf.float32, name="data", shape=[8, 32, 32, 3])
    out = tf.placeholder(dtype=tf.float32, name="out", shape=[8, 32, 32, 8])

    # as the structure of the simple model
    def model():
        conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME', name="conv")
        out = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases), name="relu")

    # saver is used for saving model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model()
        # initialize all variables
        tf.global_variables_initializer().run()
        # save the model in the file path
        saver.save(sess, './model')
```

在上面这份代码中，我们生成了一个输入大小是8\*32\*32\*3，输出尺寸是8\*32\*32\*8，卷积核大小是5\*5，输入通道数是3，输出通道数是8的只有一个卷积层的神经网络。我们甚至没有给这个神经网络进行训练，就直接进行了保存。这样一个神经网络模型文件里是一个什么样的结构呢？是像一般的程序一样，编译完了程度就被转化成了一堆逻辑指令呢？还是继续维持着这样一个模型文件的结构呢？

在运行完上面的代码后，在当前路径下应该多了四个文件，checkpoint，model.index， model.meta， model.data-00000-of-00001，如果没有的话请重新运行文件生成这样的文件。这样四个文件里面保存了什么呢？按照官方文档的解释，meta文件保存了序列化后的计算图，index文件则保存了数据文件的索引，包括变量名等等，data文件是保存了类似于变量值这样的数据的文件，checkpoint记录了最新的模型文件的名称或者序号。差是所期望的标准差，但是均值是零的一个正态分布。在加法结点中，我们采取对数据加上所期望的均值的方法将用于初始化的正态分布由标准正态分布转化成一个标准差是输入的stddev参数规定的标准差，均值是输入的mean参数规定的均值的一个正态分布。

## ONNX模型结构

在之前几节的内容中，我们没有借助任何框架的内容，而是用了python或者Julia去手写网络，去训练得到结果。现在假设我们已经保存了训练的结果，也就是说，我们保存了训练得到的计算图和对应的权重，那我们如何把训练得到的模型和主流框架中的模型结构统一起来呢？也就是说，让我们训练得到的结果可以直接放在主流框架里使用呢？在这里我们选择ONNX作为复用的轮子。

```python
def make_node(
        op_type,  # type: Text
        inputs,  # type: Sequence[Text]
        outputs,  # type: Sequence[Text]
        name=None,  # type: Optional[Text]
        doc_string=None,  # type: Optional[Text]
        domain=None,  # type: Optional[Text]
        **kwargs  # type: Any
):  # type: (...) -> NodeProto
    """Construct a NodeProto.

    Arguments:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for NodeProto
        doc_string (string, default None): optional documentation string for NodeProto
        domain (string, default None): optional domain for NodeProto.
            If it's None, we will just use default domain (which is empty)
        **kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.
    """
```


先来看Conv部分下的example：

```python
x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                [5., 6., 7., 8., 9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.],
                [20., 21., 22., 23., 24.]]]]).astype(np.float32)
W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

# Convolution with padding
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[1, 1, 1, 1],
)
y_with_padding = np.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                             [33., 54., 63., 72., 51.],
                             [63., 99., 108., 117., 81.],
                             [93., 144., 153., 162., 111.],
                             [72., 111., 117., 123., 84.]]]]).astype(np.float32)
expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
       name='test_basic_conv_with_padding')

# Convolution without padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[0, 0, 0, 0],
)
y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                [99., 108., 117.],
                                [144., 153., 162.]]]]).astype(np.float32)
expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
       name='test_basic_conv_without_padding')
```

这是一个pad为1，步长为1，卷积核大小是3的卷积层，输入是数据(X)和对应的卷积核(W)，输出(Y)。需要注意的是，这里的x,w,y均是名称，对应的数据是通过名称输入到卷积层中进行操作的。


# 模型的部署

下面我们以实际生成的mnist模型为例，来看一下如何在应用中集成模型进行推理。

这里以Windows平台为例，应用程序使用C#语言。我们有两种方案：

- 使用[Windows Machine Learning](https://docs.microsoft.com/zh-cn/windows/ai/)加载模型并推理，这种方式要求系统必须是Windows 10，版本号大于17763
- 使用由微软开源的[OnnxRuntime](https://github.com/Microsoft/onnxruntime)库加载模型并推理，系统可以是Windows 7 或 Windows 10，但目前仅支持x64平台

---

## 使用[Windows Machine Learning](https://docs.microsoft.com/zh-cn/windows/ai/)

### 创建UWP项目
### 设计界面

打开`MainPage.xaml`，将整个Grid片段替换为如下代码：

``` xml
<Grid>
    <StackPanel VerticalAlignment="Center" HorizontalAlignment="Center">
        <Grid x:Name="inkGrid" Background="Black" Width="336" Height="336">
            <InkCanvas x:Name="inkCanvas" />
        </Grid>

        <TextBlock x:Name="lbResult" FontSize="30"/>

        <Button x:Name="btnClean" Content="clean" FontSize="30" Tapped="btnClean_Tapped"/>
    </StackPanel>
</Grid>
```

- `inkCanvas`是写数字的画布，由于训练mnist用的数据集是黑色背景白色字，所以这里将画布也设置为黑色背景。这个控件无法直接设置背景色，所以在外面包一层背景为黑色的`Grid`
- `lbResult`文本控件用来显示识别的结果
- `btnClean`按钮用来清除之前的画布

在`MainPage`构造函数中调用`InitInk`方法初始化画布，设置画笔颜色为白色：

``` C#
private void InitInk()
{
    inkCanvas.InkPresenter.InputDeviceTypes = CoreInputDeviceTypes.Mouse | CoreInputDeviceTypes.Touch;
    var attr = new InkDrawingAttributes();
    attr.Color = Colors.White;
    attr.IgnorePressure = true;
    attr.PenTip = PenTipShape.Circle;
    attr.Size = new Size(24, 24);
    inkCanvas.InkPresenter.UpdateDefaultDrawingAttributes(attr);

    inkCanvas.InkPresenter.StrokesCollected += InkPresenter_StrokesCollected;
}

private void InkPresenter_StrokesCollected(InkPresenter sender, InkStrokesCollectedEventArgs args)
{
    RecogNumberFromInk();
}
```

其中，`RecogNumberFromInk`就是对画布中的内容进行识别，在后面的小节中再添加实现。

添加`btnClean`按钮事件的实现：

``` C#
private void btnClean_Tapped(object sender, TappedRoutedEventArgs e)
{
    inkCanvas.InkPresenter.StrokeContainer.Clear();
    lbResult.Text = string.Empty;
}
```

### 画布数据预处理

打开生成的模型调用封装`mnist.cs`查看输入格式：

``` C#
public sealed class mnistInput
{
    public TensorFloat port; // shape(784)
}
```

可以知道该模型需要大小为784的float数组，对应的是28\*28大小图片的每个像素点的色值。

这里添加函数将画布渲染到bitmap，然后按模型的要求组织784大小的float数组：

``` C#
public async Task<float[]> GetInputDataFromInk()
{
    // 将画布渲染到大小为28*28的bitmap上，与模型的输入保持一致
    RenderTargetBitmap renderBitmap = new RenderTargetBitmap();
    await renderBitmap.RenderAsync(inkGrid, 28, 28);

    // 取出所有像素点的色值
    var buffer = await renderBitmap.GetPixelsAsync();
    var imageBytes = buffer.ToArray();

    float[] data = new float[784];
    for (int i = 0; i < 784; i++)
    {
        // 画布为黑白色的，可以直接取RGB中的一个分量作为此像素的色值
        int baseIndex = 4 * i;
        data[i] = imageBytes[baseIndex];
    }

    return data;
}
```

### 调用模型进行推理

整理好输入数据后，就可以调用模型进行推理并输出结果了。前面界面设计部分，`inkCanvas`添加了响应事件并调用了`RecogNumberFromInk`方法，这里给出对应的实现：

``` C#
private async void RecogNumberFromInk()
{
    // 从文件加载模型
    var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/mnist.onnx"));
    var model = await mnistModel.CreateFromStreamAsync(modelFile);

    // 组织输入
    var inputArray = await GetInputDataFromInk();
    var inputTensor = TensorFloat.CreateFromArray(new List<long> { 784 }, inputArray);
    var modelInput = new mnistInput { port = inputTensor };

    // 推理
    var result = await model.EvaluateAsync(modelInput);

    // 得到每个数字的得分
    var scoreList = result.dense3port.GetAsVectorView().ToList();

    // 从输出中取出得分最高的
    var max = scoreList.IndexOf(scoreList.Max());

    // 显示在控件中
    lbResult.Text = max.ToString();
}
```


## 使用[OnnxRuntime](https://github.com/Microsoft/onnxruntime)
### 创建WPF项目
### 添加模型文件到项目中
### 添加OnnxRuntime库
### 设计界面
打开`MainWindow.xaml`，将整个Grid片段替换为如下代码：

``` xml
<Grid>
    <StackPanel>
        <Grid Width="336" Height="336">
            <InkCanvas x:Name="inkCanvas" Width="336" Height="336" Background="Black"/>
        </Grid>

        <TextBlock x:Name="lbResult" FontSize="26" HorizontalAlignment="Center"/>

        <Button x:Name="btnClean" Content="Clean" Click="BtnClean_Click" FontSize="26"/>
    </StackPanel>
</Grid>
```


- `inkCanvas`是写数字的画布，由于训练mnist用的数据集是黑色背景白色字，所以这里将画布也设置为黑色背景
- `lbResult`文本控件用来显示识别的结果
- `btnClean`按钮用来清除之前的画布

在`MainWindow`构造函数中调用`InitInk`方法初始化画布，设置画笔颜色为白色：

``` C#
private void InitInk()
{
    // 将画笔改为白色
    var attr = new DrawingAttributes();
    attr.Color = Colors.White;
    attr.IgnorePressure = true;
    attr.StylusTip = StylusTip.Ellipse;
    attr.Height = 24;
    attr.Width = 24;
    inkCanvas.DefaultDrawingAttributes = attr;

    // 每次画完一笔时，都触发此事件进行识别
    inkCanvas.StrokeCollected += InkCanvas_StrokeCollected;
}

private void InkCanvas_StrokeCollected(object sender, InkCanvasStrokeCollectedEventArgs e)
{
    // 从画布中进行识别
    RecogNumberFromInk(); 
}
```

其中，`RecogNumberFromInk`就是对画布中的内容进行识别，在后面的小节中再添加实现。

添加`btnClean`按钮事件的实现：

``` C#
private void BtnClean_Click(object sender, RoutedEventArgs e)
{
    // 清除画布
    inkCanvas.Strokes.Clear();
    lbResult.Text = string.Empty;
}
```

### 画布数据预处理

按照前面的介绍使用`Netron`打开mnist模型可以看到，输入`port`是一个大小784的float数组，对应的是28\*28大小图片的每个像素点的色值，输出`dense3port`是1\*10的float数组，分别代表识别为数字0-9的得分，值最大的即为识别结果。

因此这里需要添加几个函数对画布数据进行处理，转为模型可以接受的数据。以下几个函数分别是将画布渲染到28\*28的图片，读取每个像素点的值，生成模型需要数组：

``` C#
private BitmapSource RenderToBitmap(FrameworkElement canvas, int scaledWidth, int scaledHeight)
{
    // 将画布渲染到bitmap上
    RenderTargetBitmap rtb = new RenderTargetBitmap((int)canvas.Width, (int)canvas.Height, 96d, 96d, PixelFormats.Default);
    rtb.Render(canvas);

    // 调整bitmap的大小为28*28，与模型的输入保持一致
    TransformedBitmap tfb = new TransformedBitmap(rtb, new ScaleTransform(scaledWidth / rtb.Width, scaledHeight / rtb.Height));
    return tfb;
}

public byte[] GetPixels(BitmapSource source)
{
    if (source.Format != PixelFormats.Bgra32)
        source = new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

    int width = source.PixelWidth;
    int height = source.PixelHeight;
    byte[] data = new byte[width * 4 * height];

    source.CopyPixels(data, width * 4, 0);
    return data;
}

public float[] GetInputDataFromInk()
{
    var bitmap = RenderToBitmap(inkCanvas, 28, 28);
    var imageBytes = GetPixels(bitmap);

    float[] data = new float[784];
    for (int i = 0; i < 784; i++)
    {
        // 画布为黑白色的，可以直接取RGB中的一个分量作为此像素的色值
        int baseIndex = 4 * i;
        data[i] = imageBytes[baseIndex];
    }

    return data;
}
```

### 调用模型进行推理

整理好输入数据后，就可以调用模型进行推理并输出结果了。前面界面设计部分，`inkCanvas`添加了响应事件并调用了`RecogNumberFromInk`方法，这里给出对应的实现：

``` C#
private void RecogNumberFromInk()
{
    // 从画布得到输入数组
    var inputData = GetInputDataFromInk();

    // 从文件中加载模型
    string modelPath = AppDomain.CurrentDomain.BaseDirectory + "mnist.onnx";

    using (var session = new InferenceSession(modelPath))
    {
        // 支持多个输入，对于mnist模型，只需要一个输入
        var container = new List<NamedOnnxValue>();

        // 输入是大小784的一维数组
        var tensor = new DenseTensor<float>(inputData, new int[] { 784 });

        // 输入的名称是port
        container.Add(NamedOnnxValue.CreateFromTensor<float>("port", tensor));

        // 推理
        var results = session.Run(container);

        // 输出结果是IReadOnlyList<NamedOnnxValue>，支持多个输出，对于mnist模型，只有一个输出
        var result = results.FirstOrDefault()?.AsTensor<float>()?.ToList();
        
        // 从输出中取出得分最高的
        var max = result.IndexOf(result.Max());

        // 显示在控件中
        lbResult.Text = max.ToString();
    }
}
```

至此，所有的代码就完成了，按`F5`即可开始调试运行。

### 添加缺失的库

当前0.1.5版本的OnnxRuntime库存在个小问题，编译的时候有几个库文件没有复制到执行目录下，运行时会出现如下错误：

<img src="./media/6/error.png" />

此时需要手动将对应的库文件复制过去。这几个文件在NuGet包目录下，打开解决方案资源管理器，展开引用，在`Microsoft.ML.OnnxRuntime`上点右键，属性，找到`路径`属性，打开对应的文件夹，向上找到`Microsoft.ML.OnnxRuntime.0.1.5`目录，打开`runtimes\win10-x64\native`目录，将`mkldnn.dll`和`onnxruntime.dll`复制到程序运行目录即可。

<img src="./media/6/dll.png" />
