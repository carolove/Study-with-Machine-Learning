# 理解mlir的affine_map
参考[多面体编译技术](https://www.cnblogs.com/wujianming-110117/p/17281946.html)
![image](https://github.com/carolove/Study-with-Machine-Learning/assets/834467/6953c365-fec8-413f-9890-3e39c86c4547)

```
矩阵计算中常用的关于内存的tiling操作，也可以通过affine_map表示。通过#tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>可以表示如下图所示的tiling切分
原始为二维for结构
for from 0 to d0 step 1
  for from 0 to d1 step 1

变换为
for from 0 to d0 step 128
  for from 0 to d1 step 256
    for from 0 to 128 step 1
      for from 0 to 256 step 1

```
