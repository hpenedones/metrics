require 'torch'
metrics = require 'metrics'
gfx = require 'gfx.js'

resp = torch.DoubleTensor { -0.9, -0.8, -0.8, -0.5, -0.1, 0.0, 0.2, 0.2, 0.51, 0.74, 0.89}
labels = torch.IntTensor  { -1, -1 , 1, -1, -1, 1, 1, -1, -1, 1, 1 }

roc_points = metrics.ROC.points(resp, labels)
area = metrics.ROC.area(roc_points)

assert(area >=0 and area <= 1, "area under ROC should be in [0,1]")

print(roc_points)
print(area)

gfx.chart(roc_points)


resp = torch.load('resp_1.dat')
labels = torch.load('labels.dat')

roc_points = metrics.ROC.points(resp, labels)
area = metrics.ROC.area(roc_points)

assert(area >=0 and area <= 1, "area under ROC should be in [0,1]")

print(roc_points)
print(area)

gfx.chart(roc_points)




