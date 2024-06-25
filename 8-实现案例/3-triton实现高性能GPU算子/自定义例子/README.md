# tips
```
这是图像处理中卷积的意思。一般来说需要一个卷积核，和输入数据分别做卷积操作，像是一个函数。
torch.nn.functional.conv2d(input,weight,bias=None,stride=1,padding=0,dilation=1,groups=1)

重点需要注意的是groups和input以及weight的关系：

input——-输入tensor大小（minibatch，in_channels，iH, iW）

weight——权重大小，就是卷积核（out_channels,$\frac{in_channels}{groups}$ , kH, kW）

注意上述的的out_channels,需要时groups的倍数。groups是将input在in_channels的维度上分成sub-input的数量。这样的好处是加快卷积的速度。比如说input shape=(8, 32, 100, 100),那么如果group设置为8，那么就是将input分成8个sub-stack array。每个array的维度是(8, 4, 100, 100)。

那么我们设置卷积核weight的时候，out_channels应该是groups的倍数。如果你嫌弃这个设计比较麻烦，groups可以设置为1.

注意：权重参数中，第一个卷积核的输出通道数，第二个是输入通道数
```
