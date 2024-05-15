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

记录一下自己整理的非整除tile size 实际切分处理过程的计算；
1. 输入###input:    
tensor<40x60x90xf32>
###weight:  
tensor<40x90x80xf32>
###out:        
tensor<40x60x80xf32>

2. 预设TileSize
我给的Tile size: (1,    11,    9，  32)

3. Processor InfonumTiles :
numTiles : (nprocs) =(size - offset).ceilDiv(tile_size)
d2=(80 - 0 ).ceilDiv( 9 ) =9
d1=(60 - 0 ).ceilDiv(11 ) = 6
d0=(40 - 0 ).ceilDiv( 1 ) = 40

splitDim =splitDim.floorDiv(numTiles)
dimValue : （procId)=(splitDim % numTiles)
d2=( s0 % 9 )
d1=((s0 floordiv 9) mod 6)
d0=((s0 floordiv 9) floordiv 6)

实际Tile结果的IR如下：
d2:
splitDim :%workgroup_id_x = hal.interface.workgroup.id[0] : index
dimValue : %7=affine.apply affine_map<()[s0] -> (s0 mod 9)>()[%workgroup_id_x]
numTiles : %6 = affine.apply affine_map<() -> (9)>()

d1:
splitDim : %8=affine.apply affine_map<()[s0] -> (s0 floordiv 9)>()[%workgroup_id_x]
dimValue : %10 =  affine.apply 
affine_map<()[s0] -> ((s0 floordiv 9) mod 6)>()[%workgroup_id_x]
numTiles :%9 = affine.apply affine_map<() -> (6)>()

d0:
splitDim :%11 = affine.apply 
affine_map<()[s0] -> ((s0 floordiv 9) floordiv 6)>()[%workgroup_id_x]
dimValue : %11 = affine.apply 
affine_map<()[s0] -> ((s0 floordiv 9) floordiv 6)>()[%workgroup_id_x]
numTiles : %12 = affine.apply affine_map<() -> (40)>()

4. Distribute Info
每个维度的边界值计算：
Value lb = loopRange.offset;
Value ub = loopRange.size;
Value step = tileSizeVals[index];

分块结果计算：
numWgroups=(ub-lb).ceilDiv(step)    -- (s1 - s0).ceilDiv(s2)
lb_partitioned = lb + procId * step   --- {s0 + s1 * s2} {lb , procId, step}
step_partitioned = step * nprocs    -----{s0 * s1} {step, nprocs}
minMap = (tileSize , ub - lb )    ---------{s0, s1 - d0} {lb, tileSize, ub}

5. 举个栗子
以d1维度为例：
distributeLB param：
lb: 0 ;
procId : %10 = affine.apply affine_map<()[s0] -> ((s0 floordiv 9) mod 6)>()[%workgroup_id_x]
step : %9 = affine.apply affine_map<() -> (6)>()

计算公式 ：{lb + procId * step}
result： 
%15 = affine.apply 
affine_map<()[s0] -> ((s0 floordiv 9) * 11 - ((s0 floordiv 9) floordiv 6) * 66)>()[%workgroup_id_x]
size param:
lb : %15 = affine.apply affine_map<()[s0] -> ((s0 floordiv 9) * 11 - ((s0 floordiv 9) floordiv 6) * 66)>()[%workgroup_id_x] 
tileSize: %c11 = arith.constant 11 : index
ub :  %c60 = arith.constant 60 : index

计算公式 ：{s0, s1 - d0}
result：
%17 = affine.min 
affine_map<()[s0] -> (11, (s0 floordiv 9) * -11 + ((s0 floordiv 9) floordiv 6) * 66 + 60)>()[%wg_id_x]

大概画一下实际映射关系：

![image](https://github.com/carolove/Study-with-Machine-Learning/assets/834467/c84c7c7f-8278-4dee-85d3-8f236c0e07fb)

