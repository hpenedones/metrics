Torch metrics package
====================

This package provides utility functions to evaluate your machine learning models.


#### Receiver Operator Curves (ROC)

Used to evalute performance of binary classifiers, and their trade-offs in terms of false-positive and false-negative rates.

```lua
require 'torch'
mt = require 'metrics'
gfx = require 'gfx.js'

resp = torch.DoubleTensor(6)
resp[1] = -0.3
resp[2] = -0.2
resp[3] = 0.0
resp[4] = 0.5
resp[5] = 0.6
resp[6] = -0.1

labels = torch.IntTensor(6)
labels[1] = -1
labels[2] = 1
labels[3] = -1
labels[4] = 1
labels[5] = 1
labels[6] = -1

roc_points = mt.roc.curve(resp, labels)
area = mt.roc.area(roc_points)

print(roc_points)

print(area)

gfx.chart(roc_points)
```


#### Confusion matrix

Used to evaluate performance of multi-class classifiers.