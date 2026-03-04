torch_roc
=========

This package provides utility functions to compute Receiver Operator Curves (ROC) for your machine learning models.

#### Disclaimer:

Use at your own risk. The code is not extensively tested, therefore it might contain bugs. 
If you find any, please let me know and I will try to fix it.

#### Installation:

```sh
git clone https://github.com/hpenedones/torch_roc.git
cd torch_roc
luarocks make
```

#### Receiver Operator Curves (ROC)

Used to evalute performance of binary classifiers, and their trade-offs in terms of false-positive and false-negative rates.

```lua
require 'torch'
torch_roc = require 'torch_roc'
gfx = require 'gfx.js'

resp = torch.DoubleTensor { -0.9, -0.8, -0.8, -0.5, -0.1, 0.0, 0.2, 0.2, 0.51, 0.74, 0.89}
labels = torch.IntTensor  {   -1,   -1,    1,   -1,   -1,   1,   1,  -1,   -1,    1,    1}

roc_points, thresholds = torch_roc.roc.points(resp, labels)
area = torch_roc.roc.area(roc_points)

print(roc_points)
print(thresholds)
print(area)

gfx.chart(roc_points)

```

![](https://raw.githubusercontent.com/hpenedones/torch_roc/master/img/roc1.png)
