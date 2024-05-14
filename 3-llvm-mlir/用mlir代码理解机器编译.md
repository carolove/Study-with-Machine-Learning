# 用mlir代码理解机器编译
## model层 mlir表示
```
%0 = "tosa.conv2d"(%input, %filter, %bias)
       {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]}
     : (tensor<1x225x225x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>)
     -> tensor<1x112x112x32xf32>
```
## linalg层 mlir表示
```
eg1：
%0 = scf.for %iv0 = ... to ... step ... iter_args(...) -> (tensor<1x112x112x32xf32>) {
  %1 = scf.for ... {
    %input_slice = tensor.extract_slice ...
    %filter_slice = tensor.extract_slice ...
    %bias_slice = tensor.extract_slice ...
    %conv = linalg.conv_2d_nhwc_hwcf {...} ins(%input_slice, %filter_slice) ...
    %generic = linalg.generic ins(%conv, %bias_slice} ... {
      %add = arith.addf ...
      linalg.yield %add ...
    }
    scf.yield %generic
  }
  scf.yield %1
}
```
```
eg2:
func.func @matmul(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>,
                  %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
```
## Tiling分块、 Buffer内存映射、 distribution布局映射后的linalg层mlir表示
```
eg1：
scf.for %ivz = (%idz * %tilez) to %ubz step (%countz * %tilez) {
  scf.for ... {
    %input_subview = memref.subview ...
    %filter_subview = memref.subview ...
    %bias_subview = memref.subview ...
    %output_subview = memref.subview ...
    linalg.conv_2d_nhwc_hwcf {...}
      ins(%input_subview, %filter_subview) outs(%output_subview) ...
    linalg.generic
      ins(%output_subview, %bias_subview) outs(%ouput_subview) ... {
      %add = arith.addf ...
      linalg.yield %add ...
    }
  }
}
```
```
eg2:
(M, N) = (16, 32) 的大小进行分块：
func.func @matmul(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                  %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %dimM = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dimK = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dimN = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = scf.for %arg3 = %c0 to %dimM step %c16 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
    %1 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 16)>(%arg3)[%dimM]
    %2 = scf.for %arg5 = %c0 to %dimN step %c32 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {
      %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 32)>(%arg5)[%dimN]
      %sliceA = tensor.extract_slice %arg0[%arg3, 0] [%1, %dimK] [1, 1]...
      %sliceB = tensor.extract_slice %arg1[0, %arg5] [%dimK, %3] [1, 1]...
      %sliceC = tensor.extract_slice %arg6[%arg3, %arg5] [%1, %3] [1, 1]...
      %4 = linalg.matmul
             ins(%sliceA, %sliceB : tensor<?x?xf32>, tensor<?x?xf32>)
             outs(%sliceC : tensor<?x?xf32>) -> tensor<?x?xf32>
      %insert = tensor.insert_slice %4 into %arg6[%arg3, %arg5] [%1, %3] [1, 1]...
      scf.yield %insert : tensor<?x?xf32>
    }
    scf.yield %2 : tensor<?x?xf32>
  }
  return %0 : tensor<?x?xf32>
}

分配 (Distribution),
%idy   = gpu.thread_id y
%dimy  = gpu.block_dim y
%lby   = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%idy, %c4]
%stepy = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%dimy, %c4]
scf.for %ivy = %lby to %c8 step %stepy {
  %idx   = gpu.thread_id  x
  %dimx  = gpu.block_dim  x
  %lbx   = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%idx, %c4]
  %stepx = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%dimx, %c4]
  scf.for %arg3 = %lbx to %c32 step %stepx {
    ...
  }
}
```
## Vectorization向量化访存寄存器布局后的vector层mlir表示
```
eg1：
scf.for %ivz = (%idz * %tilez) to %ubz step (%countz * %tilez) {
  scf.for ... {
    %input_subview = memref.subview ...
    %filter_subview = memref.subview ...
    %bias_subview = memref.subview ...
    %output_subview = memref.subview ...
    vector.transfer_read %input_subview ...
    vector.transfer_read %filter_subivew ...
    ...
    %v0 = vector.fma ...
    %v1 = vector.fma ...
    ...
    vector.transfer_write %v0, %output_subview ...
    vector.transfer_write %v1, %output_subview ...
    ...
  }
}
```
## 最终可以 进一步lowering 到scf, cf dialect然后export到目标硬件
