抑制层
Dropout
基类：_DropoutNd
Dropout类
torch.nn.Dropout(p=0.5, inplace=False)
- Input: (∗). Input can be of any shape
- Output: (∗). Output is of the same shape as input
- p (float) - probability of an element to be zeroed. Default: 0.5
- inplace (bool) – If set to True, will do this operation in-place. Default: False
Examples:
  m = nn.Dropout(p=0.2)
  input = torch.randn(20, 16)
  output = m(input)
Dropout1D类
torch.nn.Dropout1d(p=0.5, inplace=False)
- p (float, optional) – probability of an element to be zero-ed.
- inplace (bool, optional) – If set to True, will do this operation in-place
Shape:
- Input: (�,�,�)(N,C,L)
- Output: (�,�,�)(N,C,L) (same shape as input).
Dropout2D类
- Input: (�,�,�,�,�)(N,C,D,H,W) or (�,�,�,�)(C,D,H,W).
- Output: (�,�,�,�,�)(N,C,D,H,W) or (�,�,�,�)(C,D,H,W) 
Dropout3D类
- Input: (�,�,�,�,�)(N,C,D,H,W) or (�,�,�,�)(C,D,H,W).
- Output: (�,�,�,�,�)(N,C,D,H,W) or (�,�,�,�)(C,D,H,W) (same shape as input).
L1/L2norm
L1(w) = Loss(y, y_pred) + λ * |w|
L2(w) = Loss(y, y_pred) + λ * ||w||^2
   input(model,l1_lamada,loss_fn,pred,y)
稀疏连接 weightcontrl 待.....
Dog 高斯差分 待...