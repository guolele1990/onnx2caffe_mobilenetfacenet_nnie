# mobilenetfacenet_nnie

#### 介绍
相关博客： https://blog.csdn.net/tangshopping/article/details/110470050 ，insgihtface 工程训练，pytorch -> onnx -> caffe ->nnie
提供转换代码，供以后使用

#### 软件架构
1. 主要是提供模型框架转换代码


#### 安装教程

无
#### 使用说明

支持op:
Conv；
ConvTranspose；
BatchNormalization；
MaxPool；
AveragePool；
Relu；
PRelu；
Sigmoid；
Dropout；
Gemm (InnerProduct only)；
Matmul；
Add；
Mul；
Reshape；
Upsample ；
Concat；
Flatten；

2020.12.03 : 比一般 onnx2caffe demo 多了两个 op : PRelu、 Matmul;

2020.12.15 : 添加 op passthrough 即 yolo v5 的 focus 层，添加了三个函数输入参数，参数代表意义看我上面的博客吧；

2021.1.23 : 添加 op : gap、relu6、hswish、h-sigmoid；

2021.3.2 : 添加 op : split( caffe里叫 slice )；

2021.3.2 : 添加 nanodet 完整转换 demo 压缩包，因为它条件过多，干脆就独立成一个demo，免得改动太大，复原麻烦，为方便各位兄弟理解代码，我把我的试验模型也附上了，你们可以根据模型图来阅读代码；
#### 参与贡献

无


#### 特技

无
