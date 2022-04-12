# Assigment
Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2021`.

2021 cs231n作业  

Course: https://cs231n.github.io/

Code Download: https://cs231n.github.io/assignments/2021/assignment2_colab.zip

Enviorment:Google colab

---

## Assigment 2

### Fully Connected Nets

**affine_forward, affine_backward, relu_forward, relu_backward、softmax_loss**可以用assignment1中的实现。


**权重初始化**

神经网络使用误差反向传播算法（梯度下降）来训练，其需要从输出向输入逐层求导，根据导数（梯度）来更新权重，然后继续下一轮迭代。如果求出的导数（梯度）过小，那么权重的更新幅度会很小，学习速度就会变慢，甚至无法收敛。训练之前需要给神经网络的权重赋初始值，初始值的选择会对导数有很大影响。

**SGD+Momentum**
[http://cs231n.github.io/neural-networks-3/#sgd]

为解决传统的梯度下降算法收敛很慢的问题。

$$v_i=\gamma v_{i-1}-\eta\triangledown{L(\theta)}$$


$\gamma$是动量参数，是一个小于1的正数，

如果这一时刻更新度 vt与上一时刻更新度 vt−1的方向相同，则会加速。反之，则会减速。加动量的优势有两点：
1. 加速收敛
2. 提高精度(减少收敛过程中的振荡)


相当于每次在进行参数更新的时候，都会将之前的速度考虑进来，每个参数在各方向上的移动幅度不仅取决于当前的梯度，还取决于过去各个梯度在各个方向上是否一致，如果一个梯度一直沿着当前方向进行更新（水平方向），那么每次更新的幅度就越来越大，如果一个梯度在一个方向上不断变化（竖直方向），那么其更新幅度就会被衰减，这样我们就可以使用一个较大的学习率，使得收敛更快，同时梯度比较大的方向就会因为动量的关系每次更新的幅度减少。



**RMSProp**
迭代公式

$$s_1=\beta_1s_1+(1-\beta_1)dw_1^2$$

$$s_2=\beta_2s_2+(1-\beta_2)dw_2^2$$

$$w_1=w_1-\alpha\frac{dw_1}{\sqrt{s_1+\epsilon}}$$

$$w_2=w_2-\alpha\frac{dw_2}{\sqrt{s_2+\epsilon}}$$

s对梯度的平方做了一次平滑。更新w时，先用梯度除以$\sqrt{s_1+\epsilon}$相当于对梯度做了一次归一化。

合并上式

$$s=\beta s+(1-\beta)dw^2$$

$$w=w-\alpha\frac{dw}{\sqrt{s+\epsilon}}$$

$\epsilon$为很小的数，一般设置为$10^{-8}$ 避免除0。$\beta$为超参，一般设置为0.999

**Adam**
迭代公式

$$v=\beta_1v+(1-beta_1)dw$$

$$s=\beta_2s+(1-beta_2)dw^2$$

$$w=w-\alpha\frac{v}{\sqrt{s+\epsilon}}$$

$\beta_1$一般设为0.9,$\beta_2$一般设置为0.999

### BatchNormalization
**batchnorm** 归一化
归一化可以避免梯度消失。

**mini-batch-mean**

$$\mu_\beta\leftarrow\frac{1}{m}\sum_{i=1}^mx_i$$  

**mini-batch-variance**
$$\sigma_\beta\leftarrow\frac{1}{m}\sum_{i=1}^m(x_i-\mu_\beta)^2$$


$x：$批输入数据
$m:$当前批输入数据大小

归一化公式

$$\hat{x}_i\leftarrow\frac{x_i-\mu_\beta}{\sqrt{\sigma_{\beta}^2+\epsilon}}$$
normalize

$$y_i\leftarrow\gamma\hat{x}_i+\beta$$
scale-and-shift

$\epsilon:$添加较小的值到方差以防止除零

$\gamma:$可训练的比例参数

$\beta：$可训练的偏差参数


BN，它沿着BatchSize这个维度做normalize，在CV模型的表现就很好，但到了NLP模型，由于BatchSize一般都较小，如果还是用BN，那效果就不好了，反之，由于channel（HiddenSize）维度很大，用LN(layernorm)的效果会很好。

**BN Backward**[https://blog.csdn.net/yuechuen/article/details/71502503] 



