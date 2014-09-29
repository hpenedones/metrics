Torch metrics package
====================

This package provides utility functions to evaluate your machine learning models.


#### Receiver Operator Curves (ROC)

Used to evalute performance of binary classifiers, and their trade-offs in terms of false-positive and false-negative rates.

```lua
require 'torch'
mt = require 'metrics'
gfx = require 'gfx.js'

resp = torch.DoubleTensor { -0.9, -0.8, -0.8, -0.5, -0.1, 0.0, 0.2, 0.2, 0.51, 0.74, 0.89}
labels = torch.IntTensor  { -1, -1 , 1, -1, -1, 1, 1, -1, -1, 1, 1 }

roc_points = mt.roc.curve(resp, labels)
area = mt.roc.area(roc_points)

print(roc_points)
print(area)

gfx.chart(roc_points)

```

![](https://raw.github.com/hpenedones/metrics/master/img/roc_1.png)

#### Confusion matrix

Used to evaluate performance of multi-class classifiers.