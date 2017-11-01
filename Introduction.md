## 1. Introduction

Deep convolutional neural networks [22, 21] have led to a series of breakthroughs for image classification [21, 49, 39]. Deep networks naturally integrate low/mid/high-level features [49] and classifiers in an end-to-end multi-layer fashion, and the “levels” of features can be enriched by the number of stacked layers (depth). Recent evidence [40, 43] reveals that network depth is of crucial importance, and the leading results [40, 43, 12, 16] on the challenging ImageNet dataset [35] all exploit “very deep” [40] models, with a depth of sixteen [40] to thirty [16]. Many other non-trivial visual recognition tasks [7, 11, 6, 32, 27] have also greatly benefited from very deep models.

## 1. 引言

深度卷积神经网络[22, 21]导致了图像分类[21, 49, 39]的一系列突破。深度网络自然地将低/中/高级特征[49]和分类器以端到端多层方式进行集成，特征的“级别”可以通过堆叠层的数量（深度）来丰富。最近的证据[40, 43]显示网络深度至关重要，在具有挑战性的ImageNet数据集上领先的结果都采用了“非常深”[40]的模型，深度从16 [40]到30 [16]之间。许多其它重要的视觉识别任务[7, 11, 6, 32, 27]也从非常深的模型中得到了极大受益。

Driven by the significance of depth, a question arises: Is learning better networks as easy as stacking more layers? An obstacle to answering this question was the notorious problem of vanishing/exploding gradients [14, 1, 8], which hamper convergence from the beginning. This problem, however, has been largely addressed by normalized initialization [23, 8, 36, 12] and intermediate normalization layers [16], which enable networks with tens of layers to start converging for stochastic gradient descent (SGD) with backpropagation [22].

在深度重要性的推动下，出现了一个问题：学些更好的网络是否像堆叠更多的层一样容易？回答这个问题的一个障碍是梯度消失/爆炸[14, 1, 8]这个众所周知的问题，它从一开始就阻碍了收敛。然而，这个问题通过标准初始化[23, 8, 36, 12]和中间标准化层[16]在很大程度上已经解决，这使得数十层的网络能通过具有反向传播的随机梯度下降（SGD）开始收敛。

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [10, 41] and thoroughly verified by our experiments. Fig. 1 shows a typical example.

![resnet_fig1.png](http://ocs628urt.bkt.clouddn.com/resnet_fig1.png)

Figure 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4.

当更深的网络能够开始收敛时，暴露了一个退化问题：随着网络深度的增加，准确率达到饱和（这可能并不奇怪）然后迅速下降。意外的是，这种下降不是由过拟合引起的，并且在适当的深度模型上添加更多的层会导致更高的训练误差，正如[10, 41]中报告的那样，并且由我们的实验完全证实。图1显示了一个典型的例子。

![resnet_fig1.png](http://ocs628urt.bkt.clouddn.com/resnet_fig1.png)

图1 20层和56层的“简单”网络在CIFAR-10上的训练误差（左）和测试误差（右）。更深的网络有更高的训练误差和测试误差。ImageNet上的类似现象如图4所示。

The degradation (of training accuracy) indicates that not all systems are similarly easy to optimize. Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time).

退化（训练准确率）表明不是所有的系统都很容易优化。让我们考虑一个较浅的架构及其更深层次的对象，为其添加更多的层。存在通过构建得到更深层模型的解决方案：添加的层是恒等映射，其他层是从学习到的较浅模型的拷贝。 这种构造解决方案的存在表明，较深的模型不应该产生比其对应的较浅模型更高的训练误差。但是实验表明，我们目前现有的解决方案无法找到与构建的解决方案相比相对不错或更好的解决方案（或在合理的时间内无法实现）。

In this paper, we address the degradation problem by introducing a deep residual learning framework. Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping. Formally, denoting the desired underlying mapping as H(x), we let the stacked nonlinear layers fit another mapping of F(x):=H(x)−x. The original mapping is recast into F(x)+x. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

在本文中，我们通过引入深度残差学习框架解决了退化问题。我们明确地让这些层拟合残差映射，而不是希望每几个堆叠的层直接拟合期望的基础映射。形式上，将期望的基础映射表示为H(x)，我们将堆叠的非线性层拟合另一个映射F(x):=H(x)−x。原始的映射重写为F(x)+x。我们假设残差映射比原始的、未参考的映射更容易优化。在极端情况下，如果一个恒等映射是最优的，那么将残差置为零比通过一堆非线性层来拟合恒等映射更容易。

The formulation of F(x)+x can be realized by feedforward neural networks with “shortcut connections” (Fig. 2). Shortcut connections [2, 33, 48] are those skipping one or more layers. In our case, the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers (Fig. 2). Identity shortcut connections add neither extra parameter nor computational complexity. The entire network can still be trained end-to-end by SGD with backpropagation, and can be easily implemented using common libraries (e.g., Caffe [19]) without modifying the solvers.

![resnet_fig1.png](http://ocs628urt.bkt.clouddn.com/resnet-fig2.png)

Figure 2. Residual learning: a building block.

公式F(x)+x可以通过带有“快捷连接”的前向神经网络（图2）来实现。快捷连接[2, 33, 48]是那些跳过一层或更多层的连接。在我们的案例中，快捷连接简单地执行恒等映射，并将其输出添加到堆叠层的输出（图2）。恒等快捷连接既不增加额外的参数也不增加计算复杂度。整个网络仍然可以由带有反向传播的SGD进行端到端的训练，并且可以使用公共库（例如，Caffe [19]）轻松实现，而无需修改求解器。

![resnet_fig1.png](http://ocs628urt.bkt.clouddn.com/resnet-fig2.png)

图2. 残差学习：构建块
