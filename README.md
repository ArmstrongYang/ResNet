# ResNet
# Deep Residual Learning for Image Recognition

## Abstract

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers——8× deeper than VGG nets [40] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.

The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

## 摘要

更深的神经网络更难训练。我们提出了一种残差学习框架来减轻网络训练，这些网络比以前使用的网络更深。我们明确地将层变为学习关于层输入的残差函数，而不是学习未参考的函数。我们提供了全面的经验证据说明这些残差网络很容易优化，并可以显著增加深度来提高准确性。在ImageNet数据集上我们评估了深度高达152层的残差网络——比VGG[40]深8倍但仍具有较低的复杂度。这些残差网络的集合在ImageNet测试集上取得了3.57%的错误率。这个结果在ILSVRC 2015分类任务上赢得了第一名。我们也在CIFAR-10上分析了100层和1000层的残差网络。

对于许多视觉识别任务而言，表示的深度是至关重要的。仅由于我们非常深度的表示，我们便在COCO目标检测数据集上得到了28%的相对提高。深度残差网络是我们向ILSVRC和COCO 2015竞赛提交的基础，我们也赢得了ImageNet检测任务，ImageNet定位任务，COCO检测和COCO分割任务的第一名。



